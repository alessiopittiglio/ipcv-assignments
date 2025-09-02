import math
import re
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


def calculate_iou(box_a, box_b):
    """Compute IoU between two quadrilateral bounding boxes."""
    x_a_min, y_a_min = np.min(box_a, axis=0).ravel()
    x_a_max, y_a_max = np.max(box_a, axis=0).ravel()
    x_b_min, y_b_min = np.min(box_b, axis=0).ravel()
    x_b_max, y_b_max = np.max(box_b, axis=0).ravel()

    inter_x_min = max(x_a_min, x_b_min)
    inter_y_min = max(y_a_min, y_b_min)
    inter_x_max = min(x_a_max, x_b_max)
    inter_y_max = min(y_a_max, y_b_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    area_a = cv2.contourArea(box_a)
    area_b = cv2.contourArea(box_b)

    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def non_max_suppression(detections, iou_threshold=0.3):
    """Suppress overlapping detections using IoU threshold."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["votes"], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)

        remaining_detections = []
        for det in detections:
            iou = calculate_iou(best["bounding_box"], det["bounding_box"])
            if iou < iou_threshold:
                remaining_detections.append(det)

        detections = remaining_detections

    return kept


class Feature:
    def __init__(self, keypoint, descriptor):
        self.position = np.array(keypoint.pt)
        self.angle = keypoint.angle
        self.size = keypoint.size
        self.descriptor = descriptor
        self.joining_vector = None


class StarModel:
    def __init__(self):
        self.features = []
        self.barycenter = None

    def add_feature(self, feature):
        self.features.append(feature)

    def compute_barycenter(self):
        if not self.features:
            return None
        points = np.array([f.position for f in self.features])
        self.barycenter = np.mean(points, axis=0)
        return self.barycenter

    def compute_joining_vectors(self):
        if self.barycenter is None:
            return None
        for feature in self.features:
            feature.joining_vector = self.barycenter - feature.position


class Accumulator:
    def __init__(self, image_shape, bin_size=2):
        self.image_height, self.image_width = image_shape
        self.bin_size = bin_size

        grid_height = math.ceil(self.image_height / self.bin_size)
        grid_width = math.ceil(self.image_width / self.bin_size)

        self.grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        self.votes_per_bin = [
            [[] for _ in range(grid_width)] for _ in range(grid_height)
        ]

    def _quantize(self, point):
        x, y = point
        bin_x = int(x / self.bin_size)
        bin_y = int(y / self.bin_size)
        return bin_y, bin_x

    def vote(self, point, match):
        x, y = point
        if not (0 <= x < self.image_width and 0 <= y < self.image_height):
            return
        bin_y, bin_x = self._quantize(point)
        self.grid[bin_y, bin_x] += 1
        self.votes_per_bin[bin_y][bin_x].append(match)

    def find_peaks(self, min_votes, nms_window_size):
        if self.grid.max() < min_votes:
            return []

        thresholded_grid = self.grid.copy()
        thresholded_grid[thresholded_grid < min_votes] = 0

        local_max = maximum_filter(thresholded_grid, size=nms_window_size)
        peaks_mask = (thresholded_grid == local_max) & (thresholded_grid > 0)
        peak_indices = np.argwhere(peaks_mask)

        peaks = []
        for bin_y, bin_x in peak_indices:
            center_x = (bin_x + 0.5) * self.bin_size
            center_y = (bin_y + 0.5) * self.bin_size

            peaks.append(
                {
                    "position": (center_x, center_y),
                    "votes": self.grid[bin_y, bin_x],
                    "contributing_matches": self.votes_per_bin[bin_y][bin_x],
                }
            )

        peaks.sort(key=lambda p: p["votes"], reverse=True)
        return peaks

    def calculate_local_density(self, window_size=21):
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number.")

        peak_value = np.max(self.grid)
        if peak_value == 0:
            return 0.0

        peak_coords = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        half_win = window_size // 2

        r_start = max(0, peak_coords[0] - half_win)
        r_end = min(self.grid.shape[0], peak_coords[0] + half_win + 1)
        c_start = max(0, peak_coords[1] - half_win)
        c_end = min(self.grid.shape[1], peak_coords[1] + half_win + 1)

        local_window = self.grid[r_start:r_end, c_start:c_end]

        local_energy = np.sum(local_window)

        window_area = local_window.size
        if window_area == 0:
            return 0.0

        local_density_score = local_energy / window_area

        return local_density_score


class SiftGhtDetector:
    def __init__(
        self,
        bin_size=6,
        num_octave_layers=3,
        ratio_threshold=0.7,
        min_votes=2,
        nms_window_size=5,
        nms_iou_threshold=0.3,
        min_match_count=4,
        min_area=1000,
        dispersion_threshold=0.1,
        use_clahe=False,
        use_structural_similarity_filter=False,
        ssim_threshold=0.3,
        adaptive_strategy=True,
        laplacian_var_threshold=2200,
        verbose=False,
    ):
        self.sift_default = cv2.SIFT_create(nOctaveLayers=num_octave_layers)
        self.sift_alternative = cv2.SIFT_create()
        self.bin_size = bin_size
        self.ratio_threshold = ratio_threshold
        self.min_votes = min_votes
        self.nms_window_size = nms_window_size
        self.nms_iou_threshold = nms_iou_threshold
        self.min_match_count = min_match_count
        self.min_area = min_area
        self.dispersion_threshold = dispersion_threshold
        self.use_clahe = use_clahe
        self.use_structural_similarity_filter = use_structural_similarity_filter
        self.ssim_threshold = ssim_threshold
        self.adaptive_strategy = adaptive_strategy
        self.laplacian_var_threshold = laplacian_var_threshold
        self.verbose = verbose

        self.clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        )

    def _preprocess_image(self, image):
        if self.clahe is None:
            return image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        return cl

    def detect_and_compute(self, image, use_alternative_strategy=False):
        if use_alternative_strategy:
            if self.verbose:
                logger.info("Using ALTERNATIVE strategy (3 octaves, no CLAHE).")
            sift_detector = self.sift_alternative
            processed = image
        else:
            if self.verbose:
                logger.info("Using DEFAULT strategy (5 octaves, with CLAHE).")
            processed = self._preprocess_image(image)
            sift_detector = self.sift_default

        keypoints, descriptors = sift_detector.detectAndCompute(processed, None)
        features = [Feature(kp, desc) for kp, desc in zip(keypoints, descriptors)]
        if self.verbose:
            logger.info(f"Detected {len(features)} features.")
        return features

    def build_model(self, model_image, use_alternative_strategy=False):
        features = self.detect_and_compute(model_image, use_alternative_strategy)
        model = StarModel()
        for feature in features:
            model.add_feature(feature)
        model.compute_barycenter()
        model.compute_joining_vectors()
        if self.verbose:
            logger.info(
                f"Model built with {len(model.features)} features. "
                f"Barycenter: {model.barycenter}."
            )
        return model

    def match_features(self, model, target_features):
        model_descriptors = np.array([f.descriptor for f in model.features])
        target_descriptors = np.array([f.descriptor for f in target_features])

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = flann.knnMatch(target_descriptors, model_descriptors, k=2)
        good_matches = []

        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue

            m, n = match_pair
            if m.distance < self.ratio_threshold * n.distance:
                good_match = cv2.DMatch(
                    _queryIdx=m.trainIdx, _trainIdx=m.queryIdx, _distance=m.distance
                )
                good_matches.append(good_match)

        if self.verbose:
            logger.info(f"Found {len(good_matches)} good matches.")
        return good_matches

    def apply_scale_rotation(self, joining_vector, scale, rotation):
        scaled_vector = scale * joining_vector
        angle_rad = np.deg2rad(rotation)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        rotated_vector = np.dot(rotation_matrix, scaled_vector)
        return rotated_vector

    def vote_for_reference_points(
        self, model, target_features, good_matches, target_image_shape
    ):
        accumulator = Accumulator(target_image_shape[:2], bin_size=self.bin_size)
        for match in good_matches:
            model_idx = match.queryIdx
            target_idx = match.trainIdx

            model_feature = model.features[model_idx]
            target_feature = target_features[target_idx]

            scale_ratio = target_feature.size / model_feature.size
            rotation_diff = target_feature.angle - model_feature.angle

            transformed_vector = self.apply_scale_rotation(
                model_feature.joining_vector, scale_ratio, rotation_diff
            )
            predicted_reference = target_feature.position + transformed_vector
            accumulator.vote(predicted_reference, match)

        return accumulator

    def estimate_affine_transform(self, model_image, matches, model, target_features):
        if len(matches) < 3:
            if self.verbose:
                logger.info("Not enough matches to compute transformation.")
            return None

        src_pts = np.float32(
            [model.features[m.queryIdx].position for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [target_features[m.trainIdx].position for m in matches]
        ).reshape(-1, 1, 2)

        M_affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
        if M_affine is None:
            return None

        h, w = model_image.shape[:2]
        corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(
            -1, 1, 2
        )
        dst = cv2.transform(corners, M_affine)
        return np.int32(dst)

    def filter_by_structural_similarity(self, detections, model_image, target_image):
        if not detections:
            return []

        model_gray = cv2.cvtColor(model_image, cv2.COLOR_BGR2GRAY)
        h, w = model_gray.shape

        kept_detections = []
        for det in detections:
            bbox = det["bounding_box"]

            dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            src_pts = np.float32(bbox).reshape(4, 2)

            transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            warped_patch = cv2.warpPerspective(target_image, transform_matrix, (w, h))
            warped_patch_gray = cv2.cvtColor(warped_patch, cv2.COLOR_BGR2GRAY)

            score = ssim(
                model_gray,
                warped_patch_gray,
                data_range=warped_patch_gray.max() - warped_patch_gray.min(),
            )

            if score >= self.ssim_threshold:
                if self.verbose:
                    logger.info(
                        f"Detection at {det['position']} PASSED structural check "
                        f"(SSIM: {score:.3f})."
                    )
                kept_detections.append(det)
            else:
                if self.verbose:
                    logger.warning(
                        f"Detection at {det['position']} REJECTED by structural check "
                        f"(SSIM: {score:.3f} < {self.ssim_threshold})."
                    )

        return kept_detections

    def detect(self, model_image, target_image):
        use_alternative = False
        if self.adaptive_strategy:
            target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(target_gray, cv2.CV_64F).var()

            if laplacian_var > self.laplacian_var_threshold:
                if self.verbose:
                    logger.warning(
                        "High Laplacian variance detected (%.2f > %.2f). "
                        "Switching to robust strategy.",
                        laplacian_var,
                        self.laplacian_var_threshold,
                    )
                use_alternative = True

        model = self.build_model(model_image, use_alternative_strategy=use_alternative)
        target_features = self.detect_and_compute(
            target_image, use_alternative_strategy=use_alternative
        )
        good_matches = self.match_features(model, target_features)

        accumulator = self.vote_for_reference_points(
            model, target_features, good_matches, target_image.shape
        )

        peaks = accumulator.find_peaks(
            min_votes=self.min_votes, nms_window_size=self.nms_window_size
        )

        score = accumulator.calculate_local_density()
        if score < self.dispersion_threshold:
            if self.verbose:
                logger.warning(
                    "Spatial dispersion score %.4f below threshold %.4f, "
                    "rejecting detection.",
                    score,
                    self.dispersion_threshold,
                )
            return [], accumulator, []

        if self.verbose:
            logger.info(f"Found {len(peaks)} peaks in accumulator.")

        bounding_boxes = []
        for peak in peaks:
            matches = peak["contributing_matches"]
            bbox = self.estimate_affine_transform(
                model_image, matches, model, target_features
            )
            if bbox is not None:
                bounding_boxes.append(
                    {
                        "position": peak["position"],
                        "votes": peak["votes"],
                        "bounding_box": bbox,
                        "corners_dict": {
                            "top_left": tuple(bbox[0][0]),
                            "top_right": tuple(bbox[1][0]),
                            "bottom_right": tuple(bbox[2][0]),
                            "bottom_left": tuple(bbox[3][0]),
                        },
                        "area": cv2.contourArea(bbox),
                    }
                )

        filtered = non_max_suppression(bounding_boxes, self.nms_iou_threshold)

        if self.verbose:
            logger.info(
                f"Detected {len(bounding_boxes)} raw instances, kept {len(filtered)} after NMS.",
            )
        if self.use_structural_similarity_filter:
            filtered = self.filter_by_structural_similarity(
                filtered, model_image, target_image
            )

        return peaks, accumulator, filtered


# ======================================================================================
#  DEBUGGING AND VISUALIZATION FUNCTIONS
# ======================================================================================


def visualize_barycenter(image, model):
    image = image.copy()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(*model.barycenter, color="black", s=60)
    plt.scatter(*model.barycenter, color="red", s=50)
    plt.show()


def visualize_joining_vectors(image, model, num_vectors=3):
    image = image.copy()

    indices = np.random.choice(len(model.features), num_vectors, replace=False)
    for i in indices:
        start_point = model.features[i].position
        joining_vector = model.features[i].joining_vector
        plt.arrow(
            start_point[0],
            start_point[1],
            joining_vector[0],
            joining_vector[1],
            color="red",
            head_width=10,
        )
        plt.scatter(*start_point, color="yellow", s=60)
        plt.scatter(*start_point, color="blue", s=10)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def format_and_print_results(all_results, scene_filename):
    print(f"Results for scene: {scene_filename}")
    print("------------------------------------")
    if not all_results:
        print("No books found.")
        return

    for book_name, instances in all_results.items():
        print(f"{book_name} - {len(instances)} instance(s) found:")
        for i, inst in enumerate(instances, 1):
            corners = inst["corners_dict"]
            area = inst["area"]
            formatted_corners = {k: tuple(map(int, v)) for k, v in corners.items()}

            print(
                f"  Instance {i} {{"
                f"top_left: {formatted_corners['top_left']}, "
                f"top_right: {formatted_corners['top_right']}, "
                f"bottom_right: {formatted_corners['bottom_right']}, "
                f"bottom_left: {formatted_corners['bottom_left']}, "
                f"area: {area:.0f}px}}"
            )

    print("\n" + "=" * 50)


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]
