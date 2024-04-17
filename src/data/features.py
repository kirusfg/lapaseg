import numpy as np
from skimage.feature import hog
import cv2
from tqdm import tqdm


def calculate_hsv_histograms(patch, bins: int = 16):
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_patch], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_patch], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_patch], [2], None, [bins], [0, 256]).flatten()
    return np.concatenate([hist_h, hist_s, hist_v])


def calculate_hog_features(patch, bins: int = 9):
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray_patch,
        orientations=bins,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
    )
    return hog_features


def calculate_loc_features(pixel: tuple[int, int], image_size: tuple[int, int]):
    return np.array([pixel[0] / image_size[0], pixel[1] / image_size[1]])


def calculate_landmarks_distance(
    pixel_location: tuple[int, int], landmarks: np.ndarray
):
    distance = np.linalg.norm(landmarks - np.array(pixel_location), axis=1)
    return distance


def extract_features(
    image: np.ndarray,
    landmarks: np.ndarray,
    hsv_patch_size: int = 16,
    hog_patch_size: int = 64,
) -> np.ndarray:
    """
    Returns the sum of two decimal numbers in binary digits.

        Parameters:
            image (np.ndarray): A numpy array containing a BGR image loaded by OpenCV
            landmarks (np.ndarray): A numpy array containing 106 landmarks for the image
            hsv_patch_size (int): Patch size for HSV extraction
            hog_patch_size (int): Patch size for HOG extraction

        Returns:
            pixel_feature_vectors (np.ndarray): A numpy array containing flattened feature vectors
            [fC, fS, fL, fD] for each of the pixels, where
                fC - color feature (HSV)
                fS - shape feature (HOG)
                fL - relative location of the pixel from the image origin
                fD - Euclidian distance from the pixel to each of the landmarks
    """
    height, width, channels = image.shape

    hsv_half_patch = hsv_patch_size // 2
    hog_half_patch = hog_patch_size // 2
    max_half_patch = max(hsv_half_patch, hog_half_patch)
    extended_image = cv2.copyMakeBorder(
        image,
        max_half_patch,
        max_half_patch,
        max_half_patch,
        max_half_patch,
        cv2.BORDER_REFLECT,
    )

    pixel_feature_vectors = np.zeros((height, width, 0))
    for n in tqdm(range(height * width)):
        i = n // width
        j = n - i * width
        hsv_patch = extended_image[
            i : i + 2 * hsv_half_patch + 1, j : j + 2 * hsv_half_patch + 1
        ]
        hog_patch = extended_image[
            i : i + 2 * hog_half_patch + 1, j : j + 2 * hog_half_patch + 1
        ]

        hsv_features = calculate_hsv_histograms(hsv_patch)
        hog_features = calculate_hog_features(hog_patch)
        loc_features = calculate_loc_features((i, j), (height, width))
        dist_features = calculate_landmarks_distance((i, j), landmarks)

        print(
            f"hsv_features: {hsv_features.shape}",
            f"hog_features: {hog_features.shape}",
            f"loc_features: {loc_features.shape}",
            f"dist_features: {dist_features.shape}",
        )

        features = np.concatenate(
            (
                hsv_features,
                hog_features,
                loc_features,
                dist_features,
            )
        )

        print(features.shape)

        if pixel_feature_vectors.shape[-1] == 0:
            pixel_feature_vectors = np.zeros((height, width, len(features)))

        pixel_feature_vectors[i, j] = features

    return pixel_feature_vectors


def extract_features_pixel(
    image: np.ndarray,
    landmarks: np.ndarray,
    x: int,
    y: int,
    hsv_patch_size: int = 16,
    hog_patch_size: int = 64,
) -> np.ndarray:
    """
    Returns the sum of two decimal numbers in binary digits.

        Parameters:
            image (np.ndarray): A numpy array containing a BGR image loaded by OpenCV
            landmarks (np.ndarray): A numpy array containing 106 landmarks for the image
            hsv_patch_size (int): Patch size for HSV extraction
            hog_patch_size (int): Patch size for HOG extraction

        Returns:
            pixel_feature_vectors (np.ndarray): A numpy array containing flattened feature vectors
            [fC, fS, fL, fD] for each of the pixels, where
                fC - color feature (HSV)
                fS - shape feature (HOG)
                fL - relative location of the pixel from the image origin
                fD - Euclidian distance from the pixel to each of the landmarks
    """
    height, width, channels = image.shape

    hsv_half_patch = hsv_patch_size // 2
    hog_half_patch = hog_patch_size // 2
    max_half_patch = max(hsv_half_patch, hog_half_patch)

    extended_image = cv2.copyMakeBorder(
        image,
        max_half_patch,
        max_half_patch,
        max_half_patch,
        max_half_patch,
        cv2.BORDER_REFLECT,
    )

    ex = x + max_half_patch
    ey = y + max_half_patch

    # Extract patches centered at (x, y)
    hsv_patch = extended_image[
        ex - hsv_half_patch : ex + hsv_half_patch + 1,
        ey - hsv_half_patch : ey + hsv_half_patch + 1,
    ]
    hog_patch = extended_image[
        ex - hog_half_patch : ex + hog_half_patch + 1,
        ey - hog_half_patch : ey + hog_half_patch + 1,
    ]
    # i = x
    # j = y
    # hsv_patch = extended_image[
    #     i : i + 2 * hsv_half_patch + 1, j : j + 2 * hsv_half_patch + 1
    # ]
    # hog_patch = extended_image[
    #     i : i + 2 * hog_half_patch + 1, j : j + 2 * hog_half_patch + 1
    # ]

    hsv_features = calculate_hsv_histograms(hsv_patch)
    hog_features = calculate_hog_features(hog_patch)
    loc_features = calculate_loc_features((x, y), (height, width))
    dist_features = calculate_landmarks_distance((x, y), landmarks)

    # print(
    #     f"hsv_features: {hsv_features.shape}",
    #     f"hog_features: {hog_features.shape}",
    #     f"loc_features: {loc_features.shape}",
    #     f"dist_features: {dist_features.shape}",
    # )

    features = np.concatenate(
        (
            hsv_features,
            hog_features,
            loc_features,
            dist_features,
        )
    )

    # print(features.shape)

    return features
