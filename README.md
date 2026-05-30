
# Real-Time Fight Detection using SlowFast R50

## About the Project

This project is a real-time fight detection system built using the SlowFast R50 action recognition model.

The idea behind this project was to explore how deep learning can be used to automatically detect violent activities in surveillance footage. Instead of analyzing individual images, the model learns motion patterns across multiple video frames, allowing it to understand actions over time.

The system processes video streams, predicts whether a fight is occurring, and displays the result in real time.

---

## How It Works

The model receives a sequence of video frames and classifies the activity into one of two categories:

* Fight
* Normal

A sliding window of 32 frames is maintained during inference. These frames are processed through the SlowFast architecture, which captures both detailed motion information and long-term temporal patterns.

To make predictions more stable, a small temporal smoothing buffer is applied before displaying the final result.

---

## Architecture

```text
Video / CCTV Stream
        │
        ▼
Frame Extraction
        │
        ▼
32 Frame Buffer
        │
        ▼
Resize + Normalization
        │
        ▼
 ┌─────────────────┐
 │ Slow Pathway    │
 │ 8 Frames        │
 └────────┬────────┘
          │
          │
 ┌────────▼────────┐
 │ Fast Pathway    │
 │ 32 Frames       │
 └────────┬────────┘
          │
          ▼
    SlowFast R50
          │
          ▼
 Classification
(Fight / Normal)
          │
          ▼
 Confidence Score
          │
          ▼
 Prediction Smoothing
          │
          ▼
 Final Output
```

---

## Dataset

The model was trained using:

* 5,500+ fight videos
* CCTV surveillance footage
* Normal activity videos

The dataset contains a variety of scenes and motion patterns to help improve robustness in real-world scenarios.

---

## Technologies Used

* Python
* PyTorch
* PyTorchVideo
* OpenCV
* NumPy
* CUDA

---

## Model Details

| Parameter    | Value        |
| ------------ | ------------ |
| Architecture | SlowFast R50 |
| Input Frames | 32           |
| Resolution   | 224 × 224    |
| Classes      | 2            |
| Framework    | PyTorch      |
| Inference    | Real-Time    |

---

## Current Features

* Real-time video processing
* Fight vs Normal classification
* Confidence visualization
* Prediction smoothing
* GPU support
* CCTV video compatibility

---

## Future Work

Some improvements that can be added in future versions:

* RTSP camera support
* Multi-camera monitoring
* Web dashboard
* Alert notifications
* Event logging
* Fight localization using person detection and tracking

---

## Demo

A demo video showing real-time inference is included in this repository.
[Watch Demo Video](./demo_vide.mp4)

---

## Note

This project was developed for learning, experimentation, and surveillance AI research. Real-world performance may vary depending on camera angle, lighting conditions, video quality, and scene complexity.
