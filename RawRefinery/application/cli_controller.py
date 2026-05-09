import torch
import numpy as np
from pathlib import Path
from platformdirs import user_data_dir
from time import perf_counter
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from RawHandler.RawHandler import RawHandler
from blended_tiling import TilingModule
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from RawRefinery.application.dng_utils import convert_color_matrix, to_dng
from RawRefinery.application.postprocessing import match_colors_linear
from RawRefinery.application.utils import can_use_gpu

from RawRefinery.application.ModelHandler import MODEL_REGISTRY, key_string

class CLIInferenceWorker:
    def __init__(self, model, model_params, device, rh, conditioning, dims, img_size=128, tile_overlap=0.25, batch_size=2):
        self.model = model
        self.model_params = model_params
        self.device = device
        self.rh = rh
        self.conditioning = conditioning
        self.dims = dims
        self.img_size = img_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def process(self, progress_callback=None):
        """
        Performs the inference and returns (img_rgb, final_denoised).
        """
        try:
            image_RGGB = self.rh.as_rggb(dims=self.dims, colorspace='lin_rec2020')
            image_RGB = self.rh.as_rgb(dims=self.dims, demosaicing_func=demosaicing_CFA_Bayer_Malvar2004, colorspace='lin_rec2020', clip=True)
            
            tensor_image = torch.from_numpy(image_RGGB).unsqueeze(0).contiguous()
            tensor_RGB = torch.from_numpy(image_RGB).unsqueeze(0).contiguous()

            full_size = [image_RGGB.shape[1], image_RGGB.shape[2]]
            tile_size = [self.img_size, self.img_size]
            overlap = [self.tile_overlap, self.tile_overlap]

            # Tiling Setup
            tiling_module = TilingModule(tile_size=tile_size, tile_overlap=overlap, base_size=full_size)
            tiling_module_rgb = TilingModule(tile_size=[s*2 for s in tile_size], tile_overlap=overlap, base_size=[s*2 for s in full_size])
            tiling_module_rebuild = TilingModule(tile_size=[s*2 for s in tile_size], tile_overlap=overlap, base_size=[s*2 for s in full_size])

            tiles = tiling_module.split_into_tiles(tensor_image).float().to(self.device)
            tiles_rgb = tiling_module_rgb.split_into_tiles(tensor_RGB).float().to(self.device)
            
            batches = torch.split(tiles, self.batch_size)
            batches_rgb = torch.split(tiles_rgb, self.batch_size)

            # Conditioning Setup
            cond_tensor = torch.as_tensor(self.conditioning, device=self.device).float().unsqueeze(0)
            cond_tensor[:, 0] /= 6400
            cond_tensor[:, 1] = 0
            cond_tensor = cond_tensor[:, 0:1]

            processed_batches = []
            
            # Determine Dtype
            dtype_map = {'mps': torch.float16, 'cuda': torch.float16, 'cpu': torch.bfloat16}
            autocast_dtype = dtype_map.get(self.device.type, torch.float32)
            
            total_batches = len(batches_rgb)
            
            # Inference Loop
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                    for i, (batch, batch_rgb) in enumerate(zip(batches, batches_rgb)):
                        if self._is_cancelled:
                            return None, None
                        
                        B = batch.shape[0]
                        # Expand conditioning to match batch size
                        curr_cond = cond_tensor.expand(B, -1)
                        
                        output = self.model(batch_rgb, curr_cond)

                        # Output processing
                        if "affine" in self.model_params:
                            output, _, _ = match_colors_linear(output, batch_rgb)
                        processed_batches.append(output.cpu())
                        
                        if progress_callback:
                            progress_callback((i + 1) / total_batches)

            # Rebuild
            tiles_out = torch.cat(processed_batches, dim=0)
            stitched = tiling_module_rebuild.rebuild_with_masks(tiles_out).detach().cpu().numpy()[0]

            torch.cuda.empty_cache()

            # Post-process blending
            blend_alpha = self.conditioning[1] / 100
            final_denoised = (stitched.transpose(1, 2, 0) * (1 - blend_alpha)) + (image_RGB.transpose(1, 2, 0) * blend_alpha)

            return image_RGB.transpose(1, 2, 0), final_denoised

        except Exception as e:
            raise e

class CLIController:
    def __init__(self):
        self.model = None
        self.rh = None
        self.iso = 100
        self.colorspace = 'lin_rec2020'
        self.device = torch.device("cpu")
        self.model_params = {}
        self.pub = serialization.load_pem_public_key(key_string.encode('utf-8'))

        # Manage devices
        devices = {
                   "cuda": can_use_gpu(),
                   "mps": torch.backends.mps.is_available(),
                   "cpu": lambda : True
        }
        available_devices = [d for d, is_available in devices.items() if is_available]
        if available_devices:
            self.set_device(available_devices[0])
        
    def set_device(self, device):
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)
        print(f"Using Device {self.device} from {device}")

    def load_rh(self, path):
        self.rh = RawHandler(path, colorspace=self.colorspace)
        if 'EXIF ISOSpeedRatings' in self.rh.full_metadata:
            self.iso = int(self.rh.full_metadata['EXIF ISOSpeedRatings'].values[0])
        else:
            self.iso = 100
        return self.iso

    def load_model(self, model_key):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Model {model_key} not found in registry.")

        conf = MODEL_REGISTRY[model_key]
        self.model_params = conf
        app_name = "RawRefinery"
        data_dir = Path(user_data_dir(app_name))
        model_path = data_dir / conf["filename"]

        if not model_path.is_file():
            if conf["url"]:
                print(f"Downloading {model_key}...")
                if not self._download_file(conf["url"], model_path):
                    raise Exception("Failed to download model.")
            else:
                raise Exception(f"Model file not found at {model_path}")

        try:
            print(f"Loading model: {model_path}")
            self._verify_model(model_path, model_path.with_suffix(f'{model_path.suffix}.sig'))
            loaded = torch.jit.load(model_path, map_location='cpu')
            self.model = loaded.eval().to(self.device)
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def process_image(self, input_path, output_path, model_key, conditioning, dims=None, progress_callback=None):
        self.load_rh(input_path)
        if not self.model:
            self.load_model(model_key)
        
        worker = CLIInferenceWorker(self.model, self.model_params, self.device, self.rh, conditioning, dims)
        img_rgb, final_denoised = worker.process(progress_callback=progress_callback)
        
        # Save as DNG
        transform_matrix = np.linalg.inv(
                self.rh.rgb_colorspace_transform(colorspace=self.colorspace)
                )

        CCM = self.rh.rgb_colorspace_transform(colorspace='XYZ')
        CCM = np.linalg.inv(CCM)

        # The denoised image is already in RGB space from the worker
        # But for to_dng, we need it in the correct colorspace and format.
        # The worker returns final_denoised in RGB.
        
        # We need to convert it back to the raw-like space if we want to save as CFA.
        # However, the original code does:
        # transformed = denoised @ transform_matrix.T
        # uint_img = np.clip(transformed * 2**16-1, 0, 2**16-1).astype(np.uint16)
        # ccm1 = convert_color_matrix(CCM)
        # to_dng(uint_img, self.rh, self.filename, ccm1, save_cfa=self.save_cfa, convert_to_cfa=True)
        
        # Let's follow that.
        # denoised in worker is already final_denoised which is RGB.
        
        transformed = final_denoised @ transform_matrix.T
        uint_img = np.clip(transformed * 2**16-1, 0, 2**16-1).astype(np.uint16)
        ccm1 = convert_color_matrix(CCM)
        
        to_dng(uint_img, self.rh, output_path, ccm1, save_cfa=True, convert_to_cfa=True)

    def _verify_model(self, dest_path, sig_path):
        try:
            data = Path(dest_path).read_bytes()
            signature = Path(sig_path).read_bytes()
            self.pub.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256(),
            )
            print(f"Model {dest_path} verified!")
            return True
        except Exception as e:
            print(e)
            if dest_path.exists():
                dest_path.unlink()
            if sig_path.exists():
                sig_path.unlink()
            print(f"Model {dest_path} not verified! Deleting.")
            return False

    def _download_file(self, url, dest_path):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            r = requests.get(url + '.sig', stream=True)
            r.raise_for_status()
            sig_path = dest_path.with_suffix(f'{dest_path.suffix}.sig')
            with open(sig_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return self._verify_model(dest_path, sig_path)
        except Exception as e:
            print(e)
            return False
