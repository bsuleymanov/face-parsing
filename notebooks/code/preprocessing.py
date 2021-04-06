from pathlib import Path
import numpy as np
import cv2
import argparse
import pandas as pd
from shutil import copyfile
import tqdm

def mkdir_if_empty_or_not_exist(dir_name):
    if (not dir_name.exists() or
        next(dir_name.iterdir(), None) is None):
        Path.mkdir(dir_name, exist_ok=True)
    else:
        raise Exception

def celeba_processing(data_dir="../../data", verbose=0):
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow',
                  'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat',
                  'ear_r', 'neck_l', 'neck', 'cloth']

    n_images = 30000
    img_size = 512
    data_dir = Path(data_dir)
    celeba_dir = data_dir / "CelebAMask-HQ"
    celeba_mask_anno_dir = celeba_dir / "CelebAMask-HQ-mask-anno"

    celeba_mask_save_dir = celeba_dir / "CelebAMask-HQ-mask"
    mkdir_if_empty_or_not_exist(celeba_mask_save_dir)

    for k in range(n_images):
        folder_num = k // 2000
        mask = np.zeros((img_size, img_size))
        for idx, label in enumerate(label_list):
            mask_part_path = (celeba_mask_anno_dir / str(folder_num) /
                              f"{str(k).rjust(5, '0')}_{label}.png")
            if Path.exists(mask_part_path):
                if verbose > 0:
                    print(label, idx + 1)
                mask_part = cv2.imread(str(mask_part_path))
                mask_part = mask_part[:, :, 0]
                mask[mask_part != 0] = (idx + 1)

        mask_save_path = celeba_mask_save_dir / f"{k}.png"
        if verbose > 0:
            print(mask_save_path)
        cv2.imwrite(str(mask_save_path), mask)


def celeba_partition(data_dir="../../data", verbose=0):
    data_dir = Path(data_dir)
    celeba_dir = data_dir / "CelebAMask-HQ"
    raw_img_dir = celeba_dir / "CelebA-HQ-img"
    raw_mask_dir = celeba_dir / "CelebAMask-HQ-mask"

    train_img_dir = celeba_dir / "train_images"
    mkdir_if_empty_or_not_exist(train_img_dir)
    train_mask_dir = celeba_dir / "train_masks"
    mkdir_if_empty_or_not_exist(train_mask_dir)
    val_img_dir = celeba_dir / "val_images"
    mkdir_if_empty_or_not_exist(val_img_dir)
    val_mask_dir = celeba_dir / "val_masks"
    mkdir_if_empty_or_not_exist(val_mask_dir)
    test_img_dir = celeba_dir / "test_images"
    mkdir_if_empty_or_not_exist(test_img_dir)
    test_mask_dir = celeba_dir / "test_masks"
    mkdir_if_empty_or_not_exist(test_mask_dir)

    train_count, val_count, test_count = 0, 0, 0
    image_list = pd.read_csv(celeba_dir / "CelebA-HQ-to-CelebA-mapping.txt",
                             delim_whitespace=True, header=None, skiprows=1)
    for idx, original_idx in tqdm.tqdm(enumerate(image_list.loc[:, 1])):
        if verbose > 0:
            print(idx, original_idx)
        if original_idx >= 162771 and original_idx < 182638:
            copyfile(raw_img_dir / f"{idx}.jpg", val_img_dir / f"{val_count}.jpg")
            copyfile(raw_mask_dir / f"{idx}.png", val_mask_dir / f"{val_count}.png")
            val_count += 1
        elif original_idx >= 182638:
            copyfile(raw_img_dir / f"{idx}.jpg", test_img_dir / f"{test_count}.jpg")
            copyfile(raw_mask_dir / f"{idx}.png", test_mask_dir / f"{test_count}.png")
            test_count += 1
        else:
            copyfile(raw_img_dir / f"{idx}.jpg", train_img_dir / f"{train_count}.jpg")
            copyfile(raw_mask_dir / f"{idx}.png", train_mask_dir / f"{train_count}.png")
            train_count += 1

    if verbose > 0:
        print(train_count + val_count + test_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="celeba",
                        choices=["celeba"], help="Name of dataset.")
    parser.add_argument("-p", "--path", type=str, default="../../data",
                        help="Path to data folder.")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        choices=[0, 1, 2], help="Output verbosity.")
    args = parser.parse_args()
    if args.dataset == "celeba":
        try:
            celeba_processing(data_dir=args.path, verbose=args.verbose)
        except:
            pass
        celeba_partition(data_dir=args.path, verbose=args.verbose)

if __name__ == "__main__":
    main()