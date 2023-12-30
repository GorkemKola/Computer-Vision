# %%
'''
Importing libraries
____________________
numpy: array operations
sklearn.svm: classifier
cv2: image operations
os: folder operations

part1.hough_transform_circles: getting circles
'''
import numpy as np
from sklearn import svm
import cv2
import os
from part1 import hough_transform_circles

# %%
def read_img(path, size=None, resized=False):
    '''
        To Read Images
        _____________

        Parameters
        _____________
        path : str
            image path
        size : int
            to resize image
        resized: bool
            if true resize
    '''
    image = cv2.imread(path)
    original_size = image.shape[0]
    ratio = 1
    if resized:
        ratio = image.shape[1] / image.shape[0]
        image = cv2.resize(image, (int(size * ratio), size))
        ratio = original_size / image.shape[0]
    return image, ratio

# %%
# creating output directories
os.makedirs('TestV_HoG', exist_ok=True)
os.makedirs('TestR_HoG', exist_ok=True)

# %%
# Reading paths and images and sizes and train labels
test_r_paths = os.listdir('./TestR/')
test_r_imgs, test_r_sizes = [], []
for path in test_r_paths:
    read_result = read_img(f'./TestR/{path}', 640, True)
    test_r_imgs.append(read_result[0])
    test_r_sizes.append(read_result[1])

test_v_paths = os.listdir('./TestV/')
test_v_imgs, test_v_sizes = [], []
for path in test_v_paths:
    read_result = read_img(f'./TestV/{path}', 640, True)
    test_v_imgs.append(read_result[0])
    test_v_sizes.append(read_result[1])

train_paths = os.listdir('./Train/')
train_imgs, train_sizes, train_labels = [], [], []
for path in train_paths:
    read_result = read_img(f'./Train/{path}', 128, True)
    train_imgs.append(read_result[0])
    train_sizes.append(read_result[1])

    path = path.split('_')
    path = path[0] + '_' + path[1]
    train_labels.append(path)

# to show the results
test_v_original = []
for path in test_v_paths:
    read_result = read_img(f'./TestV/{path}')
    test_v_original.append(read_result[0])

test_r_original = []
for path in test_r_paths:
    read_result = read_img(f'./TestR/{path}')
    test_r_original.append(read_result[0])

# %%
test_r_circles = [hough_transform_circles(img=img, 
                                          min_radius=8, 
                                          max_radius=65, 
                                          threshold=60,
                                          show_inner=False) for img in test_r_imgs]

test_v_circles = [hough_transform_circles(img=img, 
                                          min_radius=8, 
                                          max_radius=65, 
                                          threshold=60,
                                          show_inner=False) for img in test_v_imgs]

train_circles = [hough_transform_circles(img=img, 
                                          min_radius=4, 
                                          max_radius=65, 
                                          threshold=60,
                                          show_inner=False) for img in train_imgs]

# %%
def get_cropped_images(image, circles, size):
    '''
        Crop Circles from image to apply train and prediction as square then makes outside the circle 0s

        Parameters
        ___________
        image : np.array
            Image will be cropped

        circles : np.array
            Contains Y, X coordinates and Radii of each circle

        size : tuple(int, int)
            Size of image

        Returns
        __________
        found_images : np.array
            found and cutted circles
    '''
    found_images = []
    image = np.copy(image)
    for y, x, r in circles:
        y_start = max(0, y - r)
        y_end = min(image.shape[0], y + r)
        x_start = max(0, x - r)
        x_end = min(image.shape[1], x + r)
        
        cropped = image[y_start:y_end, x_start:x_end]
        # Create a mask for points outside the circle
        y_indices, x_indices = np.ogrid[y_start:y_end, x_start:x_end]
        mask = (y_indices - y)**2 + (x_indices - x)**2 - r**2 > 0

        # Apply the mask to set values outside the circle to 0
        cropped[mask] = 0
        cropped = cv2.resize(cropped, size)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        found_images.append(cropped)
    return np.array(found_images)

def save_to_hist(degrees, hist, magnitudes):
    '''
        This function save Orianted Gradient information to a histogram

        Parameters
        ___________
        degrees : np.array
            Oriantation of gradients

        magnitudes : np.array
            Magnitude of gradients

        hist : np.array
            Histogram of Orianted Gradients and Returns it
    '''
    hist_range = 180 / len(hist)
    idxs = ((degrees / hist_range) - 1/2).astype(int)
    hist[idxs%len(hist)] = np.minimum(magnitudes * (- degrees + (idxs+3/2) * hist_range) / hist_range, magnitudes)
    hist[(idxs+1) % len(hist)] += np.maximum(magnitudes * (degrees - (idxs+1/2) * hist_range) / hist_range, 0)
    return hist

def get_angle_and_magnitudes(images):
    '''
        This funcion gives Oriantation and Magnitude information of image gradient

        It first apply sobel filter to get gradients than calculates magnitude and oriantation
        Parameters
        __________
        images : int
            Image array

        Returns
        __________
        gradient_magnitudes : np.array
            magnitude of gradients 

        gradient_oriantations : np.array
            oriantation of gradients
    '''
    sobel_x = cv2.Sobel(images, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(images, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitudes = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_oriantations = np.arctan2(sobel_y, sobel_x + 1e-6)
    return gradient_magnitudes, gradient_oriantations

def get_hog(gradient_magnitudes, gradient_oriantation, patch_size):
    '''
        This function gives HoG features from magnitude and oriantation information with respect to a patch size information

        Parameters
        ___________
        gradient_magnitudes : np.array
            Magnitude of Gradients

        gradient_oriantation : np.array
            Oriantation of gradients

        Returns
        _________
        hog: np.array
            hog features of an image
    '''
    hog = []
    for y in range(0, gradient_magnitudes.shape[0]-patch_size[0], patch_size[0]):
        for x in range(0, gradient_magnitudes.shape[1]-patch_size[1], patch_size[1]):
            grad = gradient_magnitudes[y:y+patch_size[0]*2, x:x+patch_size[1]*2].flatten()
            angle = gradient_oriantation[y:y+patch_size[0]*2, x:x+patch_size[1]*2].flatten()

            hist = np.zeros(4*9)
            
            for k in range(4):
                hist[9*k:9*k+9] = save_to_hist(angle, hist[9*k:9*k+9], grad)
            
            hog.extend(hist)
    return np.array(hog)

def get_features(images, circles_arr, img_size, patch_size):
    '''
        This function give hog features of images

        It crops coins from images first then calculate hog features of each coin and return it.
        Parameters
        __________
        images : np.array
            Image array

        circles_arr : list[np.array]
            Contains Y, X coordinates and Radii of each circle in each image

        img_size : (int, int)
            resizing cropped coins

        patch_size : (int, int)
            Size of each non-overlapping patch

        Returns
        _________
        hogs : list[list[np.array]]
            List contains hog information for all coins in each image.
    '''
    coins_arr = []
    for i, image in enumerate(images):
        circles = circles_arr[i]
        coins = get_cropped_images(image, circles, img_size)
        coins_arr.append(coins)
    
    hogs = []
    for coins in coins_arr:
        gradients, magnitudes = get_angle_and_magnitudes(coins)

        hog = []
        for gradient, magnitude in zip(gradients, magnitudes):
            hog.append(get_hog(gradient, magnitude, patch_size))
        hogs.append(hog)

    return hogs

# %%
# getting hog information
test_v_hog = get_features(test_v_imgs, test_v_circles, (128, 128), (16, 16))
test_r_hog = get_features(test_r_imgs, test_r_circles, (128, 128), (16, 16))
train_hog = get_features(train_imgs, train_circles, (128, 128), (16, 16) )
train_hog = np.array(train_hog)
train_hog = train_hog.reshape(-1, train_hog.shape[-1])

# %%
# Training SVM Classifier
clf = svm.SVC(kernel='linear')
clf.fit(train_hog, train_labels)

# %%
# Prediction code for test datasets
def predict(features_arr, model):
    preds = []
    for features in features_arr:
        pred = model.predict(features)
        preds.append(pred)
    return preds

# %%
# Predictions
test_v_pred = predict(test_v_hog, clf)
test_r_pred = predict(test_r_hog, clf)

# %%
def draw_results(original_images, ratios, out_dir, preds_arr, circles_arr, paths):
    '''
        This function draws circles and writes predictions on image and then saves them.

        Parameters
        ____________
        original_images : list[np.array]
            List of images that are in original sizes

        ratios : list[int]
            ratio of original size / size of image

        out_dir : str
            Output directory

        circles_arr : list[np.array]
            contains each circle information in each image

        paths : list[str]
            output paths
    '''
    for circles, image, ratio, preds, path in zip(circles_arr, original_images, ratios, preds_arr, paths):
        output = np.copy(image)
        for circle, pred in zip(circles, preds):

            circle = (ratio * circle).astype(int)
            pos = circle[1]-50, circle[0] - circle[2]

            cv2.circle(output, (circle[1], circle[0]), circle[2], (0, 0, 255), 4)
            cv2.putText(output, pred, pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

        output_path = f"{out_dir}/{path}"
        cv2.imwrite(output_path, output)

# %%
# saving results in TestV_HoG and TestR_HoG folders.
draw_results(test_v_original, test_v_sizes, 'TestV_HoG', test_v_pred, test_v_circles, test_v_paths)
draw_results(test_r_original, test_r_sizes, 'TestR_HoG', test_r_pred, test_r_circles, test_r_paths)


