Video-Based Golf Swing Phase Detection Project

Overview

This repository contains the source code and resources for the dissertation project titled "Video-Based Golf Swing Phase Detection Using CNN-LSTM Architecture: A Deep Learning Approach on the GolfDB Dataset," submitted in partial fulfillment for the degree of Master of Science in Artificial Intelligence at the University of Stirling. The project develops a CNN-LSTM hybrid model to segment golf swings into five phases—Address, Backswing, Downswing, Impact, and Follow-through—using the GolfDB dataset. The solution aims to provide a scalable, cost-effective alternative to traditional motion capture systems, enabling real-time feedback for golfers and coaches.

The code is structured as a series of Python scripts and Jupyter notebook-compatible cells, designed to run on the Kaggle platform leveraging its GPU resources. It includes data preprocessing, model training, evaluation, and visualization components, with iterative improvements such as synthetic labeling, frame-level training, data augmentation, and class-weighted loss.

Provenance of Code

Original Code: The CNN-LSTM architecture, dataset preprocessing pipeline, training scripts, and custom dataset classes are original work developed for this project.

Third-Party Libraries:
PyTorch and Torchvision: Core deep learning framework.
OpenCV: For video frame extraction and processing.
MediaPipe: For optional pose estimation (not fully integrated).
Matplotlib and NumPy: For visualization and data handling.
Scikit-learn: For evaluation metrics.
Pandas: For data manipulation and CSV handling.
PIL: For image processing.
All libraries are publicly available and cited in the dissertation references.
Pre-trained Models: Utilizes pre-trained ResNet-18 weights from PyTorch (ImageNet).
Dataset: Uses the GolfDB dataset (subset of 160 videos, /kaggle/input/golfdb-160/GolfDB), a publicly available resource with annotated golf swing videos.
AI Assistance: Code structuring and debugging were assisted by Grok (xAI), as documented in the dissertation (Chapter 3: Ethical Considerations). Final implementations are the author's own.

No proprietary binaries or large datasets are included; the code relies on Kaggle-hosted data.

Requirements

Python 3.10+

Kaggle Notebook environment (with GPU access recommended)
Dependencies (install via pip if running locally):

pip install torch torchvision opencv-python numpy matplotlib scikit-learn pandas pillow
Kaggle dataset: golfdb-160 (upload to /kaggle/input/)

How to Build and Run
Setup Environment:
On Kaggle: Fork the notebook or create a new one and upload the golfdb-160 dataset.
Locally: Install dependencies and download the GolfDB dataset separately.

Run the Pipeline:
Execute the notebook cells sequentially:

Setup & Imports: Configures libraries and checks GPU availability.
Frame Extraction & Preprocessing: Extracts and resizes frames from videos, saves to /kaggle/working/frames.
Dataset & DataLoader: Defines a custom GolfSwingDataset with synthetic or user-provided labels.
CNN-LSTM Model Definition: Builds a hybrid CNN-LSTM architecture with ResNet-18 backbone.
Training Setup: Configures loss, optimizer, and metrics; supports synthetic labels initially.
Synthetic Phase Labels: Generates equal-segment labels as a fallback.

Extract Frames for All Videos: Processes all .mp4 files in the dataset.
Handle Variable-Length Videos: Implements padding/truncation for consistent sequence length.
Frame-Level Training: Adds precision, recall, F1-score, and early stopping.
Confusion Matrix: Evaluates per-phase classification performance.
Stronger Model & Training: Incorporates data augmentation, BiLSTM, dropout, label smoothing, and mixed precision.
Class-Weighted Loss: Fine-tunes with inverse-frequency weights for imbalance.
Inference & MP4 Export: Generates annotated videos with smoothed predictions.
Per-Class Metrics: Computes and visualizes per-phase metrics.
Artifact Packaging: Bundles models, metrics, and figures.
Annotation Template: Creates a prefilled CSV for manual labeling.
Safer Annotator: Provides an interactive annotation tool.
Label Merging: Integrates user annotations into per-frame labels.
Train with User Labels: Trains using merged labels with updated metrics.
One-Video Overfit: Validates model robustness on a single video.



Performance Grid: Compares overfit and test/validation metrics.

Training may take 10–30 minutes on GPU; adjust epochs, batch size, or learning rate as needed.


Key Outputs:

Trained Models: best_cnn_lstm.pth, best_cnn_lstm_weighted.pth, etc.

Visualizations: Confusion matrices, F1-score charts, annotated MP4 videos.

Metrics: CSV files (metrics.csv, per_class_metrics.csv, etc.) for analysis.
Annotation Templates: annotations_template.csv and annotations_user.csv for labeling.
Bundled Artifacts: Stored in /kaggle/working/bundle_for_appendix.



Reproducing Results:

Use fixed seeds (e.g., random_state=42) for reproducibility.
Dataset path: Adjust DATASET_PATH to match your environment.
Expected Metrics: Validation accuracy ~80%, F1-score ~0.60 with synthetic labels; improves with manual annotations.



Project Structure

README.md: This file.
/kaggle/working/frames/: Extracted video frames organized by video ID.

/kaggle/working/bundle_for_appendix/: Artifacts (models, metrics, figures) for the dissertation appendix.
Notebook Cells: Sequential steps from setup to evaluation and annotation.

Ethical Considerations


Dataset: Uses public GolfDB data; no sensitive information included.

AI Use: Grok (xAI) assisted in code development, as noted.
Bias: Synthetic labels and dataset limitations may affect generalization; mitigated by manual annotation options.

For issues or further details, refer to the dissertation or contact the author.
