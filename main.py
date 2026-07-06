import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, blob_doh
import skimage.filters as filters #import frangi, roberts, sobel
from skimage import morphology
from skimage.segmentation import flood


INPUT_FOLDER = os.path.join(os.path.curdir, "testcases")


def detect_walls(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    img_inv = cv2.bitwise_not(gray_roi)

    # get wall shade to identify layout
    contours, _ = cv2.findContours(img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray_roi.shape[:2], dtype=np.uint8)
    
    cv2.drawContours(mask, contours, -1, 255, 1)
    pixels_gray = img_inv[mask == 255]

    if len(pixels_gray) == 0:
        raise RuntimeError("The contour region contains no pixels")
    
    shade_counts = np.bincount(pixels_gray, minlength=256)
    top_shades_indices = np.argsort(shade_counts)[::-1]
    
    shade = top_shades_indices[0]
    print("Target shade: {}".format(shade))

    shade_tolerance = 5
    img_inv[img_inv >= shade + shade_tolerance] = 0
    img_inv[img_inv < shade - shade_tolerance] = 0
    
    _, threshold_result = cv2.threshold(
        img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # in-room detection
    blurred = cv2.GaussianBlur(threshold_result, (9, 9), 0)

    cv2.drawContours(threshold_result, contours, -1, 255, 9)
    _, threshold_result = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(threshold_result, kernel, iterations=2)

    return dilated_img


#
# input:
# - img_gray - grayscale image
# - gaussian_kernel - value for input image blur (default: 9)
# output:
# - cropped_img - cropped region of interest (roi) using floor boundaries
#
def get_floor_roi(img, gaussian_kernel=9):
    # blur and threshold image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (gaussian_kernel, gaussian_kernel), 0)
    _, img_threshold = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV);

    # floodfill image
    img_floodfill = img_threshold.copy()
    h, w = img_gray.shape[:2]
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
    return cropped_img, shifted_contour


#
# input: path - path to image to be processed
#
def process_image(path):
    if not os.path.exists(path):
        raise RuntimeError("Image not found: {}".format(path))

    img = cv2.imread(path)
    
    cropped, shifted_contour = get_floor_roi(img)
    
    floor_mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(floor_mask, [shifted_contour], -1, 255, -1)

    walls_mask = detect_walls(cropped)
    clean_rooms = cv2.subtract(floor_mask, walls_mask)

    contours, _ = cv2.findContours(clean_rooms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = np.zeros(cropped.shape[:3], np.uint8)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in sorted_contours:
        if cv2.contourArea(cnt) < 500:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), -1)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 255), 7)

    kernel_close = np.ones((15, 15), np.uint8)
    output_image = cv2.morphologyEx(output_image, cv2.MORPH_CLOSE, kernel_close)

    segmented = np.zeros(cropped.shape[:3], np.uint8)
    segmented[floor_mask == 255] = (0, 255, 0)
    segmented[walls_mask == 255] = (0, 255, 255)
    
    cv2.imshow("ROI", cropped)
    cv2.imshow("Segmented Layout", segmented)
    cv2.imshow("2d Simplified Layout", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return



def main():
    if os.path.exists(INPUT_FOLDER) != True:
        raise RuntimeError("Input directory not found: {}".format(INPUT_FOLDER))
    
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            file_abspath = os.path.join(root, file)
            print(f"File: {file_abspath}")

            process_image(file_abspath)

if __name__ == '__main__':
    main()
