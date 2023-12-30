import numpy as np
import cv2
def bilinear_interpolation(image, x, y):
    """
    Perform bilinear interpolation to estimate pixel value at non-integer coordinates.

    Parameters:
    - image: Input image as a NumPy array.
    - x: X-coordinate in non-integer value.
    - y: Y-coordinate in non-integer value.

    Returns:
    - Interpolated pixel values.
    """
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    # Ensure the coordinates are within the image boundaries
    x0, y0 = np.clip(x0, 0, image.shape[1] - 1), np.clip(y0, 0, image.shape[0] - 1)
    x1, y1 = np.clip(x1, 0, image.shape[1] - 1), np.clip(y1, 0, image.shape[0] - 1)

    # Compute interpolation weights
    w_x0 = (x1 - x).reshape(-1, 1)
    w_x1 = (x - x0).reshape(-1, 1)
    w_y0 = (y1 - y).reshape(-1, 1)
    w_y1 = (y - y0).reshape(-1, 1)

    # Bilinear interpolation
    interpolated_values = (
        w_x0 * w_y0 * image[y0, x0, :] +
        w_x1 * w_y0 * image[y0, x1, :] +
        w_x0 * w_y1 * image[y1, x0, :] +
        w_x1 * w_y1 * image[y1, x1, :]
    )

    return interpolated_values
    
def warp_perspective(image, homography_matrix, output_size):
    """
    Warp an image using a homography matrix.

    Parameters:
    - image: Input image as a NumPy array.
    - homography_matrix: Homography matrix (3x3).
    - output_size: Size of the output image (width, height).

    Returns:
    - Warped image as a NumPy array.
    """
    # Create an output grid
    y, x = np.indices((output_size[1], output_size[0]))
    output_coords = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

    # Apply the homography matrix to find corresponding pixels in the input image
    src_coords = np.dot(np.linalg.inv(homography_matrix), output_coords)
    src_coords /= src_coords[2, :]

    # Clip coordinates to the valid range
    src_coords[0, :] = np.clip(src_coords[0, :], 0, image.shape[1] - 1)
    src_coords[1, :] = np.clip(src_coords[1, :], 0, image.shape[0] - 1)

    # Interpolate pixel values using bilinear interpolation
    values = bilinear_interpolation(image, src_coords[0, :], src_coords[1, :])

    # Reshape the interpolated values back to the output image shape
    warped_image = np.zeros((output_size[1], output_size[0], 3))
    warped_image[y.flatten(), x.flatten(), :] = values.reshape((-1, 3))

    return warped_image.astype(np.uint8)

def blendingMask(height, width, barrier, smoothing_window, left_biased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothing_window / 2)
    if left_biased:
        mask[:, barrier - offset : barrier + offset + 1] = np.tile(
            np.linspace(1, 0, 2 * offset + 1), (height, 1)
        )
        mask[:, : barrier - offset] = 1
    else:
        mask[:, barrier - offset : barrier + offset + 1] = np.tile(
            np.linspace(0, 1, 2 * offset + 1), (height, 1)
        )
        mask[:, barrier + offset :] = 1

    return cv2.merge([mask, mask, mask])

def panoramaBlending(dst_img_rz, src_img_warped, width_dst, showstep=False):
    h, w, _ = dst_img_rz.shape
    smoothing_window = int(width_dst / 8)
    barrier = width_dst - int(smoothing_window / 2)

    mask1 = blendingMask(h, w, barrier, smoothing_window=smoothing_window, left_biased=True)
    mask2 = blendingMask(h, w, barrier, smoothing_window=smoothing_window, left_biased=False)

    if showstep:
        nonblend = src_img_warped + dst_img_rz
    else:
        nonblend = None

    dst_img_rz = cv2.flip(dst_img_rz, 1)
    src_img_warped = cv2.flip(src_img_warped, 1)
    
    pano = src_img_warped * mask2 + dst_img_rz * mask1
    pano = cv2.flip(pano, 1)

    return pano, nonblend

def warp_images(src_img, dst_img, homography_matrix):

    height_src, width_src, _ = src_img.shape
    height_dst, width_dst, _ = dst_img.shape

    pts_src = np.float32([
        [0, 0, 1],
        [0, height_src, 1],
        [width_src, height_src, 1],
        [width_src, 0, 1],
    ]).reshape(-1, 1, 3)

    pts_dst = np.float32([
        [0, 0, 1],
        [0, height_dst, 1],
        [width_dst, height_dst, 1],
        [width_dst, 0, 1],
    ]).reshape(-1, 1, 3)

    pts_src_ = homography_matrix @ pts_src.reshape(-1, 3).T
    pts_src_ /= pts_src_[2]
    pts_src_ = pts_src_.T
    pts_src_ = pts_src_.reshape(-1, 1, 3)

    pts = np.concatenate((pts_src_, pts_dst), axis=0)
    x_min, y_min, _ = np.int64(np.min(pts, axis=0)[0] - 0.5)
    _, y_max, _ = np.int64(np.max(pts, axis=0)[0] + 0.5)

    width_pano = width_dst - x_min

    height_pano = y_max - y_min

    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) @ homography_matrix

    src_warped = warp_perspective(src_img, translation_matrix, (width_pano, height_pano))
    dst_img_rz = np.zeros((height_pano, width_pano, 3))
    dst_img_rz[-y_min : height_dst - y_min, - x_min : width_dst - x_min] = dst_img
    dst_img_rz = dst_img_rz.astype(int)
    panorama, non_blend = panoramaBlending(dst_img_rz, src_warped, width_dst, True)

    return panorama, non_blend