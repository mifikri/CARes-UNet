import cv2

import glob
import os

for image_path in glob.glob("Train-Mask/*.png"):
    print (image_path)
    os.remove(image_path)
    # Load .png image
    # image = cv2.imread(image_path)

    # # Save .jpg image
    # cv2.imwrite(image_path.replace("png", "jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
