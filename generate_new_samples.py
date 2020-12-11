import os
from glob import glob
from pathlib import Path

import cv2

root = './data/train'
generated_folder = './data/generated'
if __name__ == '__main__':

    Path(os.path.join(generated_folder, "image")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(generated_folder, "mask")).mkdir(parents=True, exist_ok=True)
    img_files = glob(os.path.join(root, 'image', '*.png'))
    mask_files = []
    for img_path in img_files:
        basename = os.path.basename(img_path)
        mask_files.append(os.path.join(root, 'mask', basename[:-4] + '_mask.png'))

    for i in range(len(img_files)):
        img = cv2.imread(img_files[i])
        mask = cv2.imread(mask_files[i])
        cv2.imwrite(os.path.join(generated_folder, "image", 'gen' + str(i * 4) + '.png'), img)
        cv2.imwrite(os.path.join(generated_folder, "mask", 'gen' + str(i * 4) + '_mask.png'), mask)
        for j in range(3):
            print(i*3+j)
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE, img)
            cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE, mask)
            cv2.imwrite(os.path.join(generated_folder, "image", 'gen' + str(i * 4 + j + 1) + '.png'), img)
            cv2.imwrite(os.path.join(generated_folder, "mask", 'gen' + str(i * 3 + j) + '_mask.png'), mask)