# UEPNet: An Upfront Exit Prediction Framework for Early-Exit Neural Networks in Intelligent Vehicles

This repository contains the implementation code for UEPNet, a novel Input Complexity-Aware Dynamic Structured Pruning framework designed for autonomous driving applications. UEPNet eliminates the sequential decision overhead found in traditional early-exit networks by proactively determining the optimal network depth before inference begins.

The framework introduces an Upfront Exit Predictor (UEP), a lightweight module that decouples decision-making from execution. By predicting the optimal exit point upfront, UEPNet enables contiguous execution and significantly reduces runtime latency.

## Repository Structure

| File | Description |
| :--- | :--- |
| `classification.ipynb` | Implementation pipeline for classification tasks (CIFAR-10) using various backbones. |
| `detection_training.ipynb` | Training script for the Object Detection backbone (YOLOv8) on the BDD100K dataset. |
| `uep.ipynb` | Implementation and evaluation of the Upfront Exit Predictor (UEP) for object detection. |

## Sample Output - Classification (VGG-16)
![Sample Classification Output](classification_op.png)

## Sample Output - Object Detection (YOLOv8m)
![Sample Object Detection Output](objectdetection_op.pdf)

## Installation & Requirements

The project requires Python 3.8+ and a GPU-enabled environment.

Clone the repository and install the necessary dependencies, primarily PyTorch and Ultralytics.
