import json
import os
import sys
import time
import argparse


def main(args):



    
    # 1. Load the dataset


    # 2. Load the VLM Agent


    # 3. Generate the hard negative samples


    # 4. Save the hard negative samples into json file


    print("Processed Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="../dataset/dataset.json")
    parser.add_argument("--output_path", type=str, default="../dataset/new_hard_options.json")
    args = parser.parse_args()
    main(args)