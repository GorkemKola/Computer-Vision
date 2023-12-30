import numpy as np
import cv2
import os

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
def read_images(data_folder):
    images = {}
    for data in os.listdir(data_folder):
        image_folder = os.path.join(data_folder, data)
        image_list = sorted(os.listdir(image_folder), key=lambda x: int(x.split('.')[0]))
        images[data] = np.array([cv2.imread(os.path.join(image_folder, path)) for path in image_list])
    return images

def draw_keypoints(path, image, keypoints):
    '''
    This Function Draw important keypoints (x, y) on the image.
    '''
    output_image = image.copy()
    cv2.drawKeypoints(image, keypoints, output_image)

    cv2.imwrite(path, output_image)


def draw_matches(path, image1, image2, point_map, inliers=None):

    rows1, cols1, _ = image1.shape
    rows2, cols2, _ = image2.shape


    match_image = np.zeros((max(rows1, rows2), cols1 + cols2, 3), dtype='uint8')

    match_image[:rows1, :cols1] = image1
    match_image[:rows2, cols1:] = image2
    for x1, y1, x2, y2 in point_map:
        point1 = int(x1), int(y1)
        point2 = int(x2 + cols1), int(y2)
        color = BLUE
        
        if inliers:
            color = GREEN if (x1, y1, x2, y2) in inliers else RED
        
        cv2.circle(match_image, point1, 5, color, -1)
        cv2.circle(match_image, point2, 5, color, -1)
        cv2.line(match_image, point1, point2, color, 1)

    cv2.imwrite(path, match_image)
