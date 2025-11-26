# Image Processing and Computer Vision - Assignments

This repository contains the assignments for the Image Processing and Computer Vision course. The course is divided into two parts, each corresponding to a separate assignment.

## Team Members

- Alessio Pittiglio
- Parsa Mastouri Kashani

## Module 1: Product Recognition of Books

### Overview
The task is to develop a computer vision system that, given a reference image for each book, is able to identify it from one picture of a shelf. The system is required to output the number of instances, their bounding boxes, and their corresponding dimensions using traditional computer vision techniques.

### Methodology
Our solution implements a **Generalized Hough Transform (GHT)** approach combined with **SIFT** features.

**Offline Phase**
1. SIFT features are extracted from the reference book image.
2. A Star Model is built where each feature stores a vector pointing towards the object's barycenter.

**Online Phase**
1. Features are extracted from the scene image.
2. Descriptors are matched using FLANN and Lowe's ratio test.
3. To handle multiple instances of the same book, we query which model feature resembles the scene feature most.
4. Each match casts a vote in a 2D accumulator for the reference point location, while accounting for scale and rotation. 

To estimate the bounding box, an affine transformation is computed via the least squares method based on the keypoints obtained from the voting mechanism.

During the preprocessing phase, CLAHE is applied to the L channel of the LAB color space to enhance the local contrast of the image. The system also adopts an adaptive strategy based on the variance of the Laplacian. If the scene is excessively sharp or noisy, the application of CLAHE is disabled to avoid quality degradation.

Finally, to filter false positives, the detected regions are mapped back to the model reference and undergo verification using the Structural Similarity Index (SSIM).

### Results
The system successfully detected **51 out of 56** instances across all test scenes. However, its performance slightly degrades in cases involving faded book spines or glare.

## Module 2: Pet Classification

### Overview
The goal of this assignment is to implement a neural network that classifies images of 37 cat and dog breeds from the Oxford-IIIT Pet dataset. It is divided into two tasks: designing a CNN from scratch and fine-tuning a pre-trained model.

### Task 1: CNN From Scratch
We decided to develop a network with less than 1 million parameters, achieving at least 60% accuracy. A customized version of MobileNetV2 was implemented with a width multiplier of 0.5x, replacing ReLU6 with standard ReLU, resulting in a total of approximately 732,000 parameters. 

The training strategy involved the use of the AdamW optimizer, the OneCycleLR scheduler, and various regularization techniques such as label smoothing, weight decay, and data augmentation using RandomResizedCrop, ColorJitter, rotations, and flips. 

Ablation studies showed that removing batch normalization reduced performance to chance levels, while removing residual connections caused a drop to approximately 58%. Without data augmentation, severe overfitting was observed, with test accuracy dropping to 31%. Similarly, removing label smoothing resulted in a decrease of approximately 6%.

The best result achieved was an accuracy of approximately **73%** on the test set.

### Task 2: Fine-Tuning ResNet-18
The goal was to achieve approximately 90% accuracy by fine-tuning a pre-trained ResNet-18. The TorchVision ResNet-18 model with ImageNet weights was used and its final fully connected layer was replaced with a custom block composed of Dropout, a 256-unit linear layer, ReLU, Dropout, and a final linear layer with 37 output units.

Training was structured into three phases: initially, only the classifier was trained; then, Layer 4 and the classification head were unfrozen; finally, Layer 3 was also unfrozen with a lower learning rate. Additional techniques were applied, such as freezing the BatchNorm statistics in the initial layers to avoid degradation, and test-time augmentation.

The best result achieved for this second task was an accuracy of **91.2%** on the test set.

## References

- Ballard, D. H. (1981). Generalizing the Hough Transform to detect arbitrary shapes. Pattern Recognition, 13(2), 111-122.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91-110.
- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In Graphics Gems IV (pp. 474-485), Academic Press Professional.
- Prof. Lisanti's lecture slides.
