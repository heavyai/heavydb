import os
import re
import shutil
import struct
import tarfile

from argparse import ArgumentParser

output_file_suffix = "_truncated"
def truncate_dump_file(file_path: str):
    extract_path = "./dump_file_extract"
    with tarfile.open(file_path, "r") as old_dump_tar:
        old_dump_tar.extractall(path=extract_path)

    for root, _, files in os.walk(extract_path):
        for file in files:
            match = re.match("\d+\\.(\d+)\\.data", file)
            if match:
                truncate_data_file(os.path.join(root, file), int(match.group(1)))

    file_name = os.path.basename(file_path)
    with tarfile.open(f"{file_name}{output_file_suffix}", "w:gz") as new_dump_tar:
        new_dump_tar.add(extract_path, arcname="")

    shutil.rmtree(extract_path)

def truncate_data_file(file_path: str, page_size: int):
    print(f"Truncating data file: {file_path}")

    file_size = os.path.getsize(file_path)
    last_empty_header_offset = None
    with open(file_path, "r+b") as file:
        file_offet = 0
        while file_offet < file_size:
            file.seek(file_offet)
            header_size = struct.unpack('i', file.read(4))[0]

            # Capture the first free page that is followed by only free pages.
            if header_size == 0 and last_empty_header_offset is None:
                last_empty_header_offset = file_offet
            elif header_size != 0 and last_empty_header_offset is not None:
                last_empty_header_offset = None
            file_offet += page_size

        if last_empty_header_offset is not None:
            # Keep at least one free page in order to ensure that code branches that check for free pages are executed.
            new_file_size = last_empty_header_offset + page_size
            if new_file_size < file_size:
                file.truncate(new_file_size)

if __name__ == '__main__':
    parser = ArgumentParser(description=f"""Truncate the dump file at the specified path for tests. Truncation involves 
                                            removal of excess free pages that do not add to test coverage. Output 
                                            dump file is created in the current directory with a file name that matches
                                            the input file name along with a {output_file_suffix} suffix.""")
    parser.add_argument("file_path", type=str, help="Path to file file")
    args = parser.parse_args()
    truncate_dump_file(file_path=args.file_path)
