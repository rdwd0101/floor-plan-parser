import os
import cv2
import numpy as np
from skimage.filters import frangi

INPUT_FOLDER = os.path.join(os.path.curdir, "testcases")


def find_non_black_pixel(img_inv):
    h, w = img_inv.shape
    for y_idx in range(0, h):
        for x_idx in range(0, w):
            if img_inv[y_idx, x_idx] > 0:
                print("Found non-black pixel: x={}, y={}, value: {}".format(x_idx, y_idx, img_inv[y_idx, x_idx]))
                return y_idx, x_idx
                


def detect_vessels_frangi(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    img_inv = cv2.bitwise_not(gray_roi)
    
    y, x = find_non_black_pixel(img_inv)

    pixel_value = img_inv[y, x]
    threshold = 30
    print("wall value: {}, threshold: {}".format(pixel_value, pixel_value+threshold))
    
    img_inv[img_inv > pixel_value + threshold] = 0
    img_inv[img_inv < pixel_value] = 0
    
    vessels_detected = frangi(img_inv, sigmas=range(3, 11), black_ridges=True)
    
    cv2.imshow('vessels', vessels_detected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(path):
    img = cv2.imread(path)

    # blur and floodfill
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, img_threshold = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV);

    img_floodfill = img_threshold.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 0);
    
    # edge detection to find largest contour (floor contour)
    edged = cv2.Canny(img_floodfill, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    floor_contour = max(contours, key=cv2.contourArea)
    
    # using contour, crop ROI
    floor_pts = floor_contour[:, 0, :]
    x_min, y_min = floor_pts.min(axis=0)
    x_max, y_max = floor_pts.max(axis=0)
    
    cropped_img = img[y_min:y_max, x_min:x_max].copy()
    shifted_contour = floor_contour - [x_min, y_min]
    local_mask = np.zeros(cropped_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    cropped_img[local_mask == 0] = 255
    cv2.imshow('original_roi.jpg', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    detect_vessels_frangi(cropped_img)


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
