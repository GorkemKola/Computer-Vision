# %%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import jit

# %%
if __name__ == '__main__':
    def read_img(path, size):
        image = cv2.imread(path)
        ratio = image.shape[1] / image.shape[0]
        image = cv2.resize(image, (int(size * ratio), size))
        return image

    # %%
    os.makedirs('Train_Hough', exist_ok=True)
    os.makedirs('TestV_Hough', exist_ok=True)
    os.makedirs('TestR_Hough', exist_ok=True)

    # %%
    test_r_paths = os.listdir('./TestR/')
    test_r_imgs = [read_img(f'./TestR/{path}', 800) for path in test_r_paths]

    test_v_paths = os.listdir('./TestV/')
    test_v_imgs = [read_img(f'./TestV/{path}', 800) for path in test_v_paths]

    train_paths = os.listdir('./Train/')
    train_imgs = [read_img(f'./Train/{path}', 200) for path in train_paths]

# %%
@jit(nopython=True)
def fill_dp(dp, edge_pixels, min_radius, max_radius, cols, rows):
    '''
        Voting function
        _______________

        Parameters
        _______________

        dp : np.array
            Accumulator array shaped (rows, columns, max_radius - min_radius)
            It keeps votes and will be returned

        edge_pixels : np.array 
            edge coordinate array edge values greater than zero

        min_radius : int
            minimum radii of circles

        max_radius : int
            maximum radii of circles

        cols : int
            shows width of image

        rows : int
            shows height of image
        
    '''
    # Iterate through coordinates
    for y, x in edge_pixels:
        # Iterate through radius
        for r in range(min_radius, max_radius + 1):
            # Iterate through thetas with 3 degree steps
            for theta in np.arange(0, 2 * np.pi, 3*np.pi/180):
                # x = a + cos(theta) * r
                # a = x - cos(theta) * r
                a = int(x - r * np.cos(theta))

                # y = b + sin(theta) * r
                # b = y - sin(theta) * r
                b = int(y - r * np.sin(theta))
                
                if 0 <= a < cols and 0 <= b < rows:
                    dp[b, a, r - min_radius] += 1 # vote if a and b in range image sizes 
    return dp # returning accumulator array


@jit(nopython=True)
def non_max_suppression(circles, min_radius, show_inner):
    '''
        Non Max Suppression Function that removes smaller and intersecting circles
        _______________

        Parameters
        ___________

        circles : np.array
            Array that contains possible circles

        min_radius : np.array
            minumum radii of circles
        
        show_inner : bool
            if True, Show circles in bigger circles

    '''
    result_circles = []
    # sorting circles minimum radii to maximum radii ones
    circles = sorted(circles, key=lambda x: x[2])

    # Iterate throuh circles
    for i in range(len(circles)):
        y, x, r = circles[i]

        # Iterate through given circle to next circles
        for j in range(i+1,len(circles)):
            y1, x1, r1 = circles[j]

            distance = np.sqrt((x1-x)**2+(y1-y)**2)
            if show_inner:
                if distance + r1 < r:
                    continue # big intertwined circle 
                
                if distance + r < r1:
                    continue # small intertwined circle

            if distance < r1 + r + 2 * min_radius:
                break # intersecting circles

        else: # save circle if loop not breaken
            result_circles.append(circles[i])
    # returning reduced circles
    return result_circles

def hough_transform_circles(img, min_radius, max_radius, threshold, show_inner=False):
    '''
        Parameters
        ___________
        img : np.array
            BGR image

        min_radius : float
            minimum radii of circles

        max_radius : float
            maximum radii of circles

        threshold : int
            to create a circle mask, we need to decide how many lines are correlated

        show_inner : bool
            if True, Show circles in bigger circles
    ''' 
    # Converting image to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # rows and columns of image
    rows, cols = img.shape
    
    # Accumulator array
    dp = np.zeros((rows, cols, max_radius - min_radius + 1))
    
    # Apply the Sobel operator to the image
    edges = cv2.Canny(img, 50, 150)

    # Combine the results to obtain the magnitude of the gradient 
    edge_pixels = np.argwhere(edges > 0)

    # Circle Voting
    dp = fill_dp(dp, edge_pixels, min_radius, max_radius, cols, rows)  

    # Filtering circles and get circle array
    circles = np.array(np.where(dp > threshold)).T

    # Eliminating small and intersecting circles using Non-max-suppression method
    circles = np.array(non_max_suppression(circles, min_radius, show_inner))
    
    # Returning circles adding min_radius to radius
    return circles + [0, 0, min_radius]

def draw_circles(img, circles):
    '''
        This function draw circles to an image
        ____________
        
        Parameters
        ___________
        img : np.array
            image will be drawn

        circles : np.array
            circles will be drawn on the image
    '''
    image_with_circles = np.copy(img)
    for circle in circles:
        cv2.circle(image_with_circles, (circle[1], circle[0]), circle[2], (0, 0, 255), 2)
    return image_with_circles

# %%
# Drawing Circles
if __name__ == '__main__':
    test_r_circles = [hough_transform_circles(img=img, 
                                            min_radius=10, 
                                            max_radius=80, 
                                            threshold=60,
                                            show_inner=True) for img in test_r_imgs]

    test_v_circles = [hough_transform_circles(img=img, 
                                            min_radius=10, 
                                            max_radius=80, 
                                            threshold=60,
                                            show_inner=True) for img in test_v_imgs]

    train_circles = [hough_transform_circles(img=img, 
                                            min_radius=10, 
                                            max_radius=100, 
                                            threshold=60,
                                            show_inner=True) for img in train_imgs]
    # %%
    def save_imgs(circles_arr, image_arr, paths, directory):
        '''
            Image Saving Function
            _____________________

            Parameters
            _________
            circles_arr : list[np.array]
                list that contains circle arrays

            image_arr : list[np.array]
                list that contains images

            image_names : list[str]
                list that contains original image names

            directory : str
                folder name resulting images will be saved
        '''
        for circles, image, path in zip(circles_arr, image_arr, paths):
            drawing = draw_circles(image, circles)
            cv2.imwrite(f'{directory}/{path}',drawing)

    # %%
    save_imgs(test_r_circles, test_r_imgs, test_r_paths, 'TestR_Hough')
    save_imgs(test_v_circles, test_v_imgs, test_v_paths, 'TestV_Hough')
    save_imgs(train_circles, train_imgs, train_paths, 'Train_Hough/')


