"""
SUN397 Dataset Preparation

1. Download SUN397:
    curl -L -O -k http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
    unzip SUN397.tar.gz

2. (Optional) Move folder to your preferred location.

3. Specify the source directory and output directory in the script below, then run it to split the dataset into train/val.
"""

import os
import shutil
from pathlib import Path


def process_dataset(txt_file, downloaded_data_path, output_folder):
    with open(txt_file, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split("/")[:-1])[1:]
        filename = input_path.split("/")[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path[1:])
        output_file_path = os.path.join(output_class_folder, filename)
        # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
        # exit()
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")


downloaded_data_path = "path/to/downloaded/SUN/data"
process_dataset(
    "Training_01.txt", downloaded_data_path, os.path.join(downloaded_data_path, "train")
)
process_dataset(
    "Testing_01.txt", downloaded_data_path, os.path.join(downloaded_data_path, "val")
)
