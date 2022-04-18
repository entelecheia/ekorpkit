import shutil
from pathlib import Path
from ekorpkit import eKonf


def utils(**args):
    args = eKonf.to_config(args)
    subtask = args.subtask

    if subtask == "duplicate":
        duplicate_folder(args)


def duplicate_folder(args):
    dupe_factor = args.dupe_factor
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True)

    if dupe_factor > 1:
        if input_dir.is_dir():
            src = str(input_dir)
            print("Source folder: ", src)
            for i in range(dupe_factor):
                if i == 0:
                    continue
                # Destination path
                dest = output_dir / f"{input_dir.name}_{i}"
                if dest.is_dir():
                    print("Destination folder already exists, skipping.")
                else:
                    destination = shutil.copytree(src, dest)
                    print("Duplicated folder as the path:", destination)
        else:
            print("input directory does not exist")

    else:
        print("dupe_factor must be greater than 1")
