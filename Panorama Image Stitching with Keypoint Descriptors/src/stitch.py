from features import extract_features, match_keypoints
import util
from homography import ransac
from warp import warp_images
import os
import cv2
import numpy as np

def stitch_images(image1, image2, method, outdir, left_ids, right_ids, dataset_name, verbose):
    merged_ids = left_ids[0], right_ids[1]
    keypoints1, descriptors1 = extract_features(image1, method, ids=merged_ids)
    keypoints2, descriptors2 = extract_features(image2, method, ids=merged_ids)
    key_point_path = os.path.join(outdir,  method,dataset_name, 'keypoints', )
    os.makedirs(key_point_path, exist_ok=True)
    path1 = os.path.join(key_point_path, f'keypoints{left_ids[0]}-{left_ids[1]}.png')
    path2 = os.path.join(key_point_path, f'keypoints{right_ids[0]}-{right_ids[1]}.png')

    util.draw_keypoints(path1, image1, keypoints1)
    util.draw_keypoints(path2, image2, keypoints2)

    good_matches = match_keypoints(descriptors1, descriptors2)
    if verbose:
        print(f'{len(good_matches)} matches found.')
        print('----------------------------------------')
    point_map = np.array([
        [keypoints1[match.queryIdx].pt[0],
        keypoints1[match.queryIdx].pt[1],
        keypoints2[match.trainIdx].pt[0],
        keypoints2[match.trainIdx].pt[1]] for match in good_matches
    ])
    match_path = os.path.join(outdir, method, dataset_name, 'matches',  )
    os.makedirs(match_path, exist_ok=True)
    path_matches = os.path.join(match_path, f'matches{merged_ids[0]}-{merged_ids[1]}.png')
    util.draw_matches(path_matches, image1, image2, point_map)
    homography, inliers = ransac(point_map, 750, 2)
    if verbose:
        print(f'{len(inliers)} inliers found')
        print('----------------------------------------')
    inlier_path = os.path.join(outdir, method, dataset_name, 'inliers', )
    os.makedirs(inlier_path, exist_ok=True)
    path_information = os.path.join(inlier_path, f'info{merged_ids[0]}-{merged_ids[1]}.txt')
    with open(path_information, 'w') as f:
        f.write('Homography Matrix:\n')
        f.write(str(homography) + '\n')
        f.write(f'Number of inliers: {len(inliers)}\n')
        f.write('Inliers: # (SourceX, SourceY, DestinationX, DestinationY)\n ' + str(inliers))

    path_inliers = os.path.join(inlier_path, f'inliers{merged_ids[0]}-{merged_ids[1]}.png')
    util.draw_matches(path_inliers, image1, image2, point_map, inliers)

    pano, non_blend = warp_images(image1, image2, homography)
    non_blend_path = os.path.join(outdir, method, dataset_name,'non_blend', )
    os.makedirs(non_blend_path, exist_ok=True)
    path_non_blend = os.path.join(non_blend_path, f'non_blend{merged_ids[0]}-{merged_ids[1]}.png')
    cv2.imwrite(path_non_blend, non_blend)

    pano = pano.astype(np.uint8)
    pano_path = os.path.join(outdir, method,dataset_name, 'panorama',  )
    os.makedirs(pano_path, exist_ok=True)
    path_pano = os.path.join(pano_path, f'panorama{merged_ids[0]}-{merged_ids[1]}.png')
    cv2.imwrite(path_pano, pano)
    return pano, merged_ids

def stitch_all(images, dataset_name, method, outdir, verbose=True, id=1):
    if len(images) == 0:
        return None, (id, id)
    if len(images) == 1:
        return images[0], (id, id)

    mid = len(images) // 2
    left_images = images[:mid]
    right_images = images[mid:]

    # Recursive calls for left and right halves
    left_pano, left_ids = stitch_all(images=left_images, 
                                     dataset_name=dataset_name, 
                                     outdir=outdir, 
                                     method=method, 
                                     id=id)
    
    right_pano, right_ids = stitch_all(right_images, 
                                       dataset_name=dataset_name, 
                                       outdir=outdir, 
                                       method=method, 
                                       id=id + len(left_images))

    # Merge step
    final_pano, merged_ids = stitch_images(left_pano, 
                                           right_pano, 
                                           method, 
                                           outdir=outdir,
                                           left_ids=left_ids, 
                                           right_ids=right_ids, 
                                           dataset_name=dataset_name, 
                                           verbose=verbose)

    return final_pano, merged_ids