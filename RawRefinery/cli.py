import argparse
import os
import glob
from pathlib import Path
from tqdm import tqdm
from RawRefinery.application.cli_controller import CLIController

def main():
    parser = argparse.ArgumentParser(description="Raw Refinery CLI tool for batch image processing.")
    parser.add_argument("--input", required=True, help="Input file or directory containing raw images.")
    parser.add_argument("--output", required=True, help="Output file or directory for processed images.")
    parser.add_argument("--model", default="Tree Net Denoise", help="Model to use (from MODEL_REGISTRY).")
    parser.add_argument("--iso", type=int, help="ISO value (will use EXIF if not provided).")
    parser.add_argument("--grain", type=int, default=0, help="Grain value (0-100).")
    parser.add_argument("--device", help="Device to use (cuda, mps, cpu).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    args = parser.parse_args()

    controller = CLIController()
    if args.device:
        controller.set_device(args.device)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Determine input files
    if input_path.is_dir():
        extensions = ['*.cr2', '*.cr3', '*.nef', '*.arw', '*.dng']
        files = []
        for ext in extensions:
            files.extend(glob.glob(str(input_path / ext)))
            files.extend(glob.glob(str(input_path / ext.upper())))
        files.sort()
    elif input_path.is_file():
        files = [str(input_path)]
    else:
        print(f"Error: Input path {input_path} does not exist.")
        return

    if not files:
        print("No compatible raw files found.")
        return

    # Determine output directory
    if output_path.is_dir():
        out_dir = output_path
    elif output_path.is_file():
        out_dir = output_path.parent
    else:
        # If it's a file path that doesn't exist, assume it's a file
        out_dir = output_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    for file_path_str in files:
        file_path = Path(file_path_str)
        output_file = output_path / f"{file_path.stem}_denoised.dng" if output_path.is_dir() else output_path

        if output_file.exists() and not args.overwrite:
            print(f"Skipping {file_path.name}, output exists. Use --overwrite to overwrite.")
            continue

        print(f"Processing {file_path.name}...")
        
        try:
            # Load ISO from file if not provided
            if args.iso is None:
                controller.load_rh(str(file_path))
                iso = controller.iso
            else:
                iso = args.iso

            conditioning = [iso, args.grain]
            
            # For CLI, we might want to process the whole image, so dims=None
            with tqdm(total=100, desc=f"{file_path.name}", bar_format="{l_bar}{{bar}}| {{n_fmt}}/{{total_fmt}} [{elapsed}<{remaining}, {rate}]") as pbar:
                controller.process_image(
                    str(file_path), 
                    str(output_file), 
                    args.model, 
                    conditioning, 
                    progress_callback=lambda v: pbar.update(int(v * 100) - pbar.n)
                )
            print(f"Successfully processed {file_path.name} -> {output_file.name}")
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    main()
