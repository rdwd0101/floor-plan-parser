# floor-plan-parser

Python/OpenCV pipeline to approximate 3d image of floor plan to simplified approximated layout.

Input: directory with images.

Output: layouts with recognized walls (yellow color) and rooms (green color), along with approximated room coordinates (x, y, height, width).

There are two types of layouts:
- segmented view, which  respectful colors on input image
- simplified 2d view, which approximates top-down floor layout

## Dependencies

OpenCV (headless) and Numpy

## Techniques used

Pipeline consists of following steps:

- Determine ROI (region of interest) - during this step, algorithm takes input image and uses preprocessing along with edge detection (Canny) to determine floor layout boundaries, then returns ROI with contour coordinates
- Detect walls - during this step, algorithm takes ROI with floor layout and firstly analyses edge of the contour, calculating the prevailing greyscale shade to filter wall boundaries (with selected shade tolerance). Then, threshold with contour parsing is applied to determine all wall-like pixels. Finally, dilation is applied in order to form a mask that is returned from the step to be used later
- The combination of floor mask and wall mask is then used to perform final post-processing step, where each one is substracted and used for contour filtering. Each contour and it's approximation is then drawn into respective layout (segmented and simplified) using colours of choice (yellow for walls and green for rooms). Approximated room coordinates is saved into JSON document in the artifacts directory.

## Launch via Docker

Clone the repository and navigate to it via terminal.

Put your sample images inside of "testcases" directory (create one if it does not exist). It will be propagated to Docker image.

Run following command to build image:

```
docker-compose build
```

Run parsing:

```
docker-compose run floor-plan-parser
```

When processing is completed, there will be subdirectories of format "output_yyyy-mm-dd_hh-mm-ss" inside of the "output" directory.

Inside of these subdirs, there will be directories corresponding to each input image with saved artifacts:

- roi.png - Region of interest (floor layout);
- segmented_layout.png - image with segmented view (same scene, but walls and room boundaries are segmented with colors);
- simplified_2d_layout.png - image with simplified (2d top-down) view;
- simplified_data.json - JSON file with metainformation regarding each detected room, format is following:
    - rooms - list of rooms (each room represent a JSON object);
        - room_name - name of the room (currently format is room_N, where N is a number assigned during parsing);
        - coordinates - x, y, width and height of room box on the ROI;
    - rooms_count - count of parsed rooms;

## License
MIT license
