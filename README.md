# ARGUS: Traffic Monitoring & Violation Tracking

**Status: Under Development**

## Run Dual-Model Video Inference

The CPU pipeline loads the 14-class vehicle/pedestrian model and the helmet/license-plate model, runs their ByteTrack instances concurrently for every frame, and shows both sets of tracked boxes live. Green boxes are vehicle/pedestrian detections; orange boxes are helmet/license-plate detections. Press `q` or `Esc` to stop early.

Create the environment and install dependencies:

```bash
uv venv --python 3.11
uv sync
```

Set `VIDEO_PATH` near the top of `dual_model_video.py`, then run:

```bash
uv run python dual_model_video.py
```

The script directly downloads the two supplied Kaggle Hub models. Kaggle downloads require credentials configured through the Kaggle API. The annotated video is saved to `output/tracked_video.mp4`. Both models are loaded as RT-DETR models and run on the GPU when CUDA is available, falling back to CPU otherwise.

ARGUS is an edge-focused traffic monitoring system designed to detect vehicles, license plates, and rider safety violations while maintaining persistent tracking across video streams.  
The project aims to build a modular pipeline capable of running on embedded hardware for real-time traffic observation, enforcement support, and data-driven traffic analytics.

ARGUS combines modern object detection with multi-object tracking to enable continuous monitoring of traffic activity and automated violation logging.

## Core Capabilities

Beyond detection, the system is intended to generate traffic statistics useful for Intelligent Transportation Systems (ITS), such as:

- **Vehicle counts and traffic flow rates**
- **Lane-level traffic density estimation**
- **Vehicle trajectory and movement analysis**
- **Speed estimation and congestion monitoring**
- **Safety violation detection** (e.g., helmet compliance)

These analytics support traffic planning, law enforcement, and urban mobility analysis, enabling smarter infrastructure and more efficient traffic management.

---

## Phase 2 Test Metrics

The model achieves the following performance metrics on the test dataset:

| Metric | Score | Decimal |
| :--- | :--- | :--- |
| **mAP50** | 95.27% | 0.9527 |
| **mAP50-95** | 85.55% | 0.8555 |
| **Precision** | 94.22% | 0.9422 |
| **Recall** | 92.20% | 0.9220 |

### Performance Visualization

The following graphs detail the training and evaluation performance across 14 different classes:

#### Precision-Recall Curve & Confusion Matrix
![Precision-Recall Curve](Phase2/Results/BoxPR_curve.png)
![Normalized Confusion Matrix](Phase2/Results/confusion_matrix_normalized.png)

#### Per-Class Performance
![Per-Class mAP](Phase2/Results/per_class_map_clean.png)

---

## Inference Examples

Sample predictions from the evaluation set demonstrating the model's detection capabilities across different viewpoints and traffic conditions:

![Inference Example 1](Phase2/Results/InferenceExamples/inference_eg01.jpg)
![Inference Example 2](Phase2/Results/InferenceExamples/inference_eg02.jpg)
