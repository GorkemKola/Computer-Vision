import cv2

def extract_features(image, method='SIFT', ids=None, verbose=True):
    '''
    This function Extracts Keypoints and descriptors from an image using 
    Scale Invariant Feature Transform (SIFT) or
    Speeded-Up Robust Features (SURF).

    Parameters:
        - image: (numpy.ndarray) RGB image
        - method: (str) ('SIFT' or 'SURF') feature extracting method.
        - verbose: (bool) if True, logs the situation.

    Returns:
        - keypoints: (tuple[cv2.KeyPoint]) keypoints list representing point of interest.
        - descriptors: (numpy.ndarray) Array containing the descriptors corresponding to keypoints.
    '''
    assert method=='SIFT' or method=='SURF', 'method variable have to be either SIFT or SURF.'
    if method=='SIFT':
        if verbose:
            print(f'Extracting SIFT Features... {ids}')
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

    else:
        if verbose:
            print(f'Extracting SURF Features... {ids}')
        
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image, None)

    if verbose:
        print('Keypoints and Descriptors are extracted.')
        print('----------------------------------------')
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    '''
    This function compares 2 descriptor array to match most similar keypoints.

    Parameters:
        - descriptors1: (numpy.ndarray)
        - descriptors2: (numpy.ndarray)

    Returns:
        - good_matches: (list[cv2.DMatch]) 
    '''
    # Use BFMatcher with KNN
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    return good_matches
