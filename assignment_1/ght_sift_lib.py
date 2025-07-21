import cv2
import re
import numpy as np
import matplotlib.pyplot as plt


class Feature:
    def __init__(self, keypoint, descriptor):
        self.position = keypoint.pt
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
        points = np.array([feature.position for feature in self.features])
        self.barycenter = np.mean(points, axis=0)

    def computing_joining_vector(self):
        if self.barycenter is None:
            return None
        for feature in self.features:
            feature.joining_vector = self.barycenter - np.array(feature.position)


class Accumulator:
    def __init__(self, image_shape, scale_bins=3, rotation_bins=8, scale_factor=1):
        self.scale_factor = scale_factor
        self.scale_bins = scale_bins
        self.rotation_bins = rotation_bins
        self.min_scale = 0.4
        self.max_scale = 0.75
        self.accumulator_size = (
            int(image_shape[0] / scale_factor),
            int(image_shape[1] / scale_factor),
            scale_bins,
            rotation_bins,
        )
        self.accumulator = np.zeros(self.accumulator_size, dtype=np.float32)

    def vote(self, predicted_reference_pixel, scale_idx, rotation_idx):
        x_scaled = predicted_reference_pixel[0] // self.scale_factor
        y_scaled = predicted_reference_pixel[1] // self.scale_factor

        h, w, s_bins, r_bins = self.accumulator_size
        if (
            0 <= x_scaled < w
            and 0 <= y_scaled < h
            and 0 <= scale_idx < s_bins
            and 0 <= rotation_idx < r_bins
        ):
            self.accumulator[y_scaled, x_scaled, scale_idx, rotation_idx] += 1

    def get_peak(self):
        max_idx = np.unravel_index(np.argmax(self.accumulator), self.accumulator.shape)
        peak_x = max_idx[1] * self.scale_factor
        peak_y = max_idx[0] * self.scale_factor
        scale_idx = max_idx[2]
        rotation_idx = max_idx[3]
        return (int(peak_x), int(peak_y), scale_idx, rotation_idx), self.accumulator[
            max_idx
        ]

    def quantize_scale_rotation(self, scale_ratio, rotation_angle):
        # Quantize scale
        scale_bin_width = (self.max_scale - self.min_scale) / self.scale_bins
        scale_idx = int((scale_ratio - self.min_scale) / scale_bin_width)
        scale_idx = max(0, min(self.scale_bins - 1, scale_idx))

        # Quantize rotation
        rotation_bin_width = 360 / self.rotation_bins
        rotation_idx = int(rotation_angle / rotation_bin_width) % self.rotation_bins

        return scale_idx, rotation_idx


class SIFT_GHT_Detector:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.model = None

    def detect_and_compute(self, image):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        features = [Feature(kp, desc) for kp, desc in zip(keypoints, descriptors)]
        return features

    def build_model(self, model_image):
        features = self.detect_and_compute(model_image)
        model = StarModel()
        for feature in features:
            model.add_feature(feature)
        model.compute_barycenter()
        model.computing_joining_vector()
        return model

    def match_features(self, model, target_features):
        model_descriptors = np.array([feature.descriptor for feature in model.features])
        target_descriptors = np.array(
            [feature.descriptor for feature in target_features]
        )

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(model_descriptors, target_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
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
        accumulator = Accumulator(target_image_shape[:2])

        for match in good_matches:
            model_idx = match.queryIdx
            target_idx = match.trainIdx

            model_feature = model.features[model_idx]
            target_feature = target_features[target_idx]

            scale_ratio = target_feature.size / model_feature.size
            rotation_diff = (target_feature.angle - model_feature.angle) % 360

            scale_idx, rotation_idx = accumulator.quantize_scale_rotation(
                scale_ratio, rotation_diff
            )

            transformed_joining_vector = self.apply_scale_rotation(
                model_feature.joining_vector, scale_ratio, rotation_diff
            )
            predicted_reference_pixel = (
                np.array(target_feature.position) + transformed_joining_vector
            )
            predicted_reference_pixel = np.round(predicted_reference_pixel).astype(
                np.int32
            )

            accumulator.vote(predicted_reference_pixel, scale_idx, rotation_idx)

        return accumulator

    def calculate_homography(self, model_image, good_matches, model, target_features):
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [model.features[m.queryIdx].position for m in good_matches]
            ).reshape(-1, 2)
            dst_pts = np.float32(
                [target_features[m.trainIdx].position for m in good_matches]
            ).reshape(-1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = model_image.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2
            )
            dst = cv2.perspectiveTransform(pts, M)
            return np.int32(dst)
        else:
            print(
                "Not enough matches are found - %d/%d"
                % (len(good_matches), MIN_MATCH_COUNT)
            )
            return None

    def detect(self, model_image, target_image):
        model = self.build_model(model_image)
        target_features = self.detect_and_compute(target_image)
        good_matches = self.match_features(model, target_features)
        accumulator = self.vote_for_reference_points(
            model, target_features, good_matches, target_image.shape
        )
        max_loc, _ = accumulator.get_peak()
        bounding_box = self.calculate_homography(
            model_image, good_matches, model, target_features
        )

        return max_loc, accumulator, bounding_box


# ======================================================================================
#  DEBUGGING AND VISUALIZATION FUNCTIONS
# ======================================================================================


def visualize_barycenter(image, model):
    image = image.copy()

    plt.imshow(image)
    plt.scatter(*model.barycenter, color="black", s=60)
    plt.scatter(*model.barycenter, color="red", s=50)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def visualize_joining_vectors(image, model, num_vectors=3):
    image = image.copy()
    for i in np.random.choice(len(model.features), num_vectors, replace=False):
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

def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]
