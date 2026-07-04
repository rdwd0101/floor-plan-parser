import os
import cv2
import numpy as np

INPUT_FOLDER = os.path.join(os.path.curdir, "testcases")


def process_image(path):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, img_threshold = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV);

    img_floodfill = img_threshold.copy()

    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(img_floodfill, mask, (0,0), 0);
    edged = cv2.Canny(img_floodfill, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Thresholded Image", img_threshold)
    cv2.imshow("Floodfilled Image", img_floodfill)
    cv2.imshow("Contoured Image", edged)
    cv2.imshow("Bounding Box Image", img)
    cv2.waitKey(0)


def main():
    if os.path.exists(INPUT_FOLDER) != True:
        raise RuntimeError("Input directory not found: {}".format(INPUT_FOLDER))
    
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for file in files:
            file_abspath = os.path.join(root, file)
            print(f"File: {file_abspath}")

            process_image(file_abspath)

if __name__ == '__main__':
    main()
