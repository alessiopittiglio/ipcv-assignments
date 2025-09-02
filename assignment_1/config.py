# ==============================================================================
# GROUND TRUTH FOR ASSIGNMENT 1 SCENES
# ==============================================================================
#
# DISCLAIMER: This is not an official ground truth dataset.
# It is a reference dataset that was manually curated on August 21, 2025
# by combining the algorithm's output with a subsequent visual inspection
# to correct errors and omissions.
#
# Purpose: To provide a stable benchmark for measuring performance and detecting
#          regressions in future algorithm changes.
#
# Format:
#   - Key: Integer index of the scene image (e.g., 0 for 'scene_0.jpg').
#   - Value: A dictionary of {book_model_name: count} or None if no books are
#            present in the scene.
#
GROUND_TRUTH = {
    0: None,
    1: {'book_18': 2},
    2: {'book_17': 1},
    3: {'book_16': 2},
    4: {'book_14': 2, 'book_15': 2},
    5: {'book_13': 1},
    6: {'book_21': 1},
    7: {'book_20': 2},
    8: None,
    9: {'book_19': 4},                  # rotated
    10: {'book_19': 4},
    11: None,                           # rotated
    12: None,                           # perspective
    13: None,
    14: None,
    15: {'book_11': 2, 'book_12': 3},   # dark
    16: {'book_11': 2, 'book_12': 3},   # B&W
    17: {'book_11': 2, 'book_12': 3},
    18: {'book_9': 3, 'book_10': 3},    # "3 x book_9 (very similar to book_8)"
    19: {'book_6': 3, 'book_7': 2},
    20: None,
    21: None,
    22: None,
    23: {'book_5': 1},
    24: None,
    25: None,
    26: {'book_0': 1, 'book_4': 1},
    27: {'book_2': 2, 'book_3': 2},
    28: {'book_1': 2}
}
