# UEPNet: Upfront Exit Prediction for Early-Exit Networks in Autonomous Driving

UEPNet is an experimental framework for reducing inference latency in early-exit deep neural networks by predicting the optimal exit point before execution. It is designed for intelligent vehicle applications and includes:

- A **classification pipeline** (CIFAR-10) demonstrating early-exit models with ResNet18, VGG16, and MobileNetV2 backbones.
- A **detection pipeline** (YOLOv8) trained on traffic-scene datasets with an early-exit branch and a final exit branch.
- An **Upfront Exit Predictor (UEP)** that learns to predict which model exit will yield the best tradeoff between accuracy and compute.
- Utilities for **latency benchmarking**, **visual explainability (Grad-CAM / LIME)**, and **deployment** via Gradio.

## Sample Outputs

**Sample output across varying driving conditions**

![Sample output across varying driving conditions](op.mp4)

**Sample model functioning on the CARLA simulator**

![Sample model functioning on the CARLA simulator](carla_op.mp4)

## Repository Structure

| File                       | Description                                                                                       |
| :------------------------- | :------------------------------------------------------------------------------------------------ |
| `requirements.txt`         | List of Python dependencies used across notebooks and scripts.                                    |
| `deployment.py`            | Gradio app to run inference using the trained UEP regressor and YOLO early/final models.          |
| `classification.ipynb`     | CIFAR-10 early-exit classification pipeline and UEP training.                                     |
| `detection_training.ipynb` | Training pipeline for YOLOv8 early/final exit models.                                             |
| `uep.ipynb`                | Evaluation of the UEP for conditional early/final YOLO inference.                                 |
| `uep_regression.ipynb`     | Regression analysis and evaluation of the UEP regressor.                                          |
| `explainability.ipynb`     | Grad-CAM and LIME explainability analysis for the YOLO early/final models and the UEP regressor.  |
| `yolo-latency.ipynb`       | Latency benchmarking for YOLO models and UEP regressor; includes YOLO model training experiments. |
| `carla.ipynb`              | CARLA simulator integration that runs conditional inference using UEP for real-time video.        |

## Dataset Sources

The notebooks and scripts in this repository operate on the following datasets; the code assumes data is available locally or via Kaggle input datasets.

- CIFAR-10 (classification)
  - https://www.cs.toronto.edu/~kriz/cifar.html
- KITTI
  - https://www.cvlibs.net/datasets/kitti/
- BDD100K (object detection)
  - https://bdd-data.berkeley.edu/

## Hardware Requirements

The code was primarily developed and tested on a **Kaggle environment with dual NVIDIA T4 GPUs**. The following estimates are based on running the full pipeline (training + evaluation):

- **GPU**: 2× NVIDIA T4 (16 GB VRAM each). For single-GPU runs, a single 16 GB GPU is sufficient for small-scale experiments, but multi-GPU is recommended for training YOLO models at scale.
- **VRAM**: Expect peak usage of ~12–14 GB per GPU during YOLO training (`yolov8m` / early-exit models) with batch sizes 32–64.
- **System RAM**: 32 GB (to accommodate large dataset loads, caching, and notebook overhead).
- **Disk**: 200+ GB free (BDD100K alone can be ~100 GB; KITTI and any additional checkpoints/models require extra space).

## Software Requirements

These requirements are derived from the imports and APIs used across the notebooks and scripts.

- **Python**: 3.8–3.11 (tested in Kaggle with 3.11). Use the same Python minor version for compatibility.
- **CUDA**: Compatible with NVIDIA T4 (CUDA 11.x; e.g., 11.7 or 11.8). Use a matching PyTorch build.

Key Python libraries (see `requirements.txt` for a pinned list):

- `torch`, `torchvision` (GPU build)
- `ultralytics` (YOLOv8)
- `numpy` (pinned below 2.0 to avoid breaking changes in some packages)
- `opencv-python`, `Pillow`
- `matplotlib`, `seaborn`
- `tqdm`
- `gradio`
- `kagglehub` (for dataset download in Kaggle notebooks)
- `pyyaml`
- `grad-cam`, `lime`, `scikit-image`
- `carla` (for CARLA simulator notebook)

## Running the Code (Step-by-Step)

### 1) Setup (conda / virtualenv / Kaggle)

1. Create a fresh environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have access to a GPU and a matching CUDA toolkit. On Kaggle, this is provided automatically.

### 2) CIFAR-10 Classification Pipeline (`classification.ipynb`)

1. Open `classification.ipynb` in a notebook environment (Kaggle, Colab, or local Jupyter).
2. Run cells sequentially.
   - The notebook downloads CIFAR-10 via `torchvision.datasets.CIFAR10`.
   - It trains early-exit versions of ResNet18, VGG16, and MobileNetV2 and then trains the UEP predictor.
3. Expected output: training progress, accuracy metrics, and exit distribution plots.

### 3) YOLO Object Detection Training (`detection_training.ipynb`)

1. This notebook assumes BDD100K data is available through a Kaggle dataset input or local download.
2. Run the notebook to prepare a `data.yaml`, preview sample images, and train an early/final YOLOv8 model.
3. Expected output: YOLO training logs, saved weights in `runs/detect/train/weights/`, and validation results.

### 4) UEP Evaluation (`uep.ipynb`)

1. Requires trained weights for:
   - `yolov8m_early_exit_dual_scale.pt` (early exit model)
   - `yolov8m_final_exit.pt` (final exit model)
2. Run the notebook to load the models and evaluate early/final predictions on the validation set. It compares IoU, precision, and recall.

### 5) Regression & Analysis (`uep_regression.ipynb`)

1. Uses the same models and datasets as `uep.ipynb`.
2. Runs additional metrics and visualization for the UEP regressor.

### 6) Explainability (`explainability.ipynb`)

1. Runs Grad-CAM (EigenCAM/GradCAM) and LIME on YOLO model outputs and the UEP regressor.
2. Saves images to `/kaggle/working/report_images` and `/kaggle/working/ppt_images`.

### 7) YOLO Latency Benchmarking (`yolo-latency.ipynb`)

1. Benchmarks inference speed for YOLO models (`yolov8m`, `yolov8l`, `yolov8x`) and the UEP regressor.
2. Uses the BDD100K dataset and the Kaggle environment.

### 8) CARLA Integration (`carla.ipynb`)

1. Requires a running CARLA simulator server (`carla.Server`).
2. Loads trained YOLO early/final models and the UEP regressor.
3. Runs a live loop capturing frames, making early/final exit decisions, recording latency, and saving annotated videos.

### 9) Deployment Script (`deployment.py`)

1. Place trained weights in the same directory or adjust paths:
   - `uep_regressor_100k.pth`
   - `yolov8m_early_100k.pt`
   - `yolov8m_final_100k.pt`
2. Run:
   ```bash
   python deployment.py
   ```
3. A Gradio app will open (or print a shareable link) where you can upload a video and get annotated output.

---
