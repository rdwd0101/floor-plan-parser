import os
import cv2
import numpy as np
import json
from datetime import datetime

INPUT_DIR = os.path.join(os.path.curdir, "testcases")
OUTPUT_DIR = os.path.join(os.path.curdir, "output")

#
# input:
#  - roi - image (3 channels)
#  - shade_tolerance - value to adjust wall shade tolerance
#  - blur_kernel_size - size for Gaussian blur kernel (default is 9, greater values allow to filter noisy room textures)
#  - contour_thickness - thickness for resulting contour drawing (relates to in-floor wall width, default is 9)
#  - dilation_kernel_size - size of kernel for final dilation step (default is 5)
# output:
# - the mask to approximate walls on the floor
#
def detect_walls(roi, shade_tolerance=5, blur_kernel_size=9, contour_thickness=9, dilation_kernel_size=5):
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

    img_inv[img_inv >= shade + shade_tolerance] = 0
    img_inv[img_inv < shade - shade_tolerance] = 0
    
    _, threshold_result = cv2.threshold(
        img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # in-room detection
    blurred = cv2.GaussianBlur(threshold_result, (blur_kernel_size, blur_kernel_size), 0)

    cv2.drawContours(threshold_result, contours, -1, 255, contour_thickness)
    _, threshold_result = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_img = cv2.dilate(threshold_result, kernel, iterations=2)

    return dilated_img


#
# input:
# - img_gray - grayscale image
# - gaussian_kernel - value for input image blur (default: 9, greater values help filter noisy input pictures)
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
# generates segmented image with walls and floor masks and saves it
#
# input:
# - img - 3-channel image
# - floor_mask - mask that represents floor approximation
# - walls_mask - mask that represents wall approximation
#
def save_segmented_layout(img, floor_mask, walls_mask, output_directory):
    segmented = np.zeros(img.shape[:3], np.uint8)
    segmented[floor_mask == 255] = (0, 255, 0)
    segmented[walls_mask == 255] = (0, 255, 255)
    
    segmented_path = os.path.join(output_directory, "segmented_layout.png")
    cv2.imwrite(segmented_path, segmented)

#
# generates simplified 2d layout image with walls and floor masks and saves it
#
# input:
# - img - 3-channel image
# - floor_mask - mask that represents floor approximation
# - walls_mask - mask that represents wall approximation
# - minimal_contour_area - value to filter small contours (default is 500, depends on size of the image)
# - morph_kernel_size - the size of morphology step (closing), default is 15, greater values provides more filled layout pictures
# - wall_thickness - width of approximated room borders (default is 7)
#
def save_simplified_2d_layout(
        img, floor_mask, walls_mask, output_directory, minimal_contour_area=500, morph_kernel_size=15, wall_thickness=7):
    rooms = cv2.subtract(floor_mask, walls_mask)
    contours, _ = cv2.findContours(rooms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = np.zeros(img.shape[:3], np.uint8)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in sorted_contours:
        if cv2.contourArea(cnt) < minimal_contour_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), -1)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 255), wall_thickness)

    kernel_close = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    output_image = cv2.morphologyEx(output_image, cv2.MORPH_CLOSE, kernel_close)
    

    simplified_2d_path = os.path.join(output_directory, "simplified_2d_layout.png")    
    cv2.imwrite(simplified_2d_path, output_image)

#
# input:
#  - path - path to the image to be processed
#  - output_subdirectory - path to store directory with output artifacts
#
def process_image(path, output_subdirectory):
    if not os.path.exists(path):
        raise RuntimeError("Image not found: {}".format(path))
    
    if not os.path.exists(output_subdirectory):
        os.makedirs(output_subdirectory)
        print("Created subdirectory: {}".format(output_subdirectory))

    # process image
    img = cv2.imread(path)    
    cropped, shifted_contour = get_floor_roi(img)
    walls_mask = detect_walls(cropped)
    
    floor_mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(floor_mask, [shifted_contour], -1, 255, -1)
    
    save_segmented_layout(cropped, floor_mask, walls_mask, output_subdirectory)
    save_simplified_2d_layout(cropped, floor_mask, walls_mask, output_subdirectory)
    
    # save region of interest
    roi_path = os.path.join(output_subdirectory, "roi.png")
    cv2.imwrite(roi_path, cropped)


def main():
    if not os.path.exists(INPUT_DIR):
        raise RuntimeError("Input directory not found: {}".format(INPUT_DIR))
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created output directory: {}".format(OUTPUT_DIR))
    
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = os.path.join(OUTPUT_DIR, "output_{}".format(timestamp))
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            file_abspath = os.path.join(root, file)
            print("Processing file: {}".format(file_abspath))
            
            image_name, _ = os.path.splitext(os.path.basename(file_abspath))
            image_name_clean = image_name.replace(" ", "").replace(".", "")
            
            output_subdirectory = os.path.join(output_directory, image_name_clean)

            process_image(file_abspath, output_subdirectory)

if __name__ == '__main__':
    main()
