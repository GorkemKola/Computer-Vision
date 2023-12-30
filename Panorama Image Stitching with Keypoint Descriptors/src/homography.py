import numpy as np
from random import sample

def compute_homography(pairs):

    M = []

    for x1, y1, x2, y2 in pairs:
        M.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        M.append([0, 0, 0, x2, y2, 1, -y2*x1, -y2*y1, -y2])

    M=np.array(M)

    _, _, V = np.linalg.svd(M)

    H = np.reshape(V[-1], (3, 3))

    H /= H.item(8)
    
    return H

def ransac(point_map, num_iterations, inlier_threshold):

    best_homography = None
    best_inliers = set()
    i = 0
    while not isinstance(best_homography, np.ndarray):
        for _ in range(num_iterations):
            indices = sample(range(len(point_map)), 4)

            sample_pair = point_map[indices]

            curr_homography = compute_homography(sample_pair)

            curr_inliers = set()

            for x1, y1, x2, y2 in point_map:
                source_point = np.array([int(x1), int(y1), 1])
                destination_point = np.array([int(x2), int(y2), 1])
                transformed_point = np.dot(curr_homography, source_point)
                error = np.linalg.norm(transformed_point - destination_point)

                if error < inlier_threshold:
                    curr_inliers.add((x1, y1, x2, y2))

            if len(curr_inliers) > len(point_map) / 50 - i and len(curr_inliers) > len(best_inliers):
                best_inliers = curr_inliers
                best_homography = curr_homography
        i += 1

    return best_homography, best_inliers
