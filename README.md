# Liver Cancer Segmentator

**Liver Cancer Segmentator: Metadata-Guided Confidence Scoring for Reliable Segmentation of Colorectal Liver Metastases in CT**

A deployment-oriented deep learning pipeline for automated segmentation of liver and colorectal liver metastases (CRLM) from contrast-enhanced CT images using **nnUNet-based 3D medical image segmentation**.

This repository provides:

* Reproducible nnUNet inference pipeline
* Automatic model downloading from Hugging Face
* Five-fold nnUNet ensemble inference
* Multi-class segmentation of liver and tumor regions
* Containerized deployment architecture for AI inference services

The trained model checkpoints are available through Hugging Face:

**Model repository**

https://huggingface.co/Hamghalam/liver-cancer-segmentation-nnunet

---

# Overview

Accurate segmentation of colorectal liver metastases on CT images is important for quantitative assessment, treatment planning, and AI-assisted radiology workflows.

This project implements a 3D nnUNet segmentation framework trained on a multi-institutional cohort of colorectal liver metastasis patients.

The model performs multi-class segmentation:

| Label | Structure                           |
| ----- | ----------------------------------- |
| 0     | Background                          |
| 1     | Liver                               |
| 2     | Tumor (colorectal liver metastasis) |

---

# Repository Structure

```
liverCancerSegmentator/

├── src/
│   ├── inference.py
│   └── model_loader.py
│
├── scripts/
│   └── download_model.py
│
├── models/
│   └── nnUNet model weights
│
├── demo/
│   ├── ct_slice.png
│   └── example_result.png
│
├── docker/
│   ├── api/
│   └── inference/
│
├── docker-compose.yml
│
├── requirements.txt
│
└── README.md
```

---

# Model Architecture

The segmentation model is based on:

* **Framework:** nnUNet v1
* **Architecture:** 3D U-Net
* **Configuration:** 3D Full Resolution
* **Training strategy:** Five-fold cross-validation ensemble
* **Deep learning framework:** PyTorch

During inference, predictions from all five trained folds are combined to improve robustness and generalization.

Pipeline:

```
                 CT Image
                     |
          nnUNet preprocessing
                     |
      --------------------------------
      |       |       |       |       |
    fold0   fold1   fold2   fold3   fold4
      --------------------------------
                     |
          Ensemble prediction
                     |
          Liver + Tumor mask
```

---

# Installation

## Clone repository

```bash
git clone https://github.com/hamghalam/liverCancerSegmentator.git

cd liverCancerSegmentator
```

## Create Python environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Download Model

The trained nnUNet checkpoints are hosted on Hugging Face.

Download the model:

```bash
python scripts/download_model.py
```

After downloading:

```
models/

└── model/
    └── nnUNetTrainerV2__nnUNetPlansv2.1/
        |
        ├── plans.pkl
        |
        ├── fold_0/
        ├── fold_1/
        ├── fold_2/
        ├── fold_3/
        └── fold_4/
```

---

# Inference

Prepare CT images in NIfTI format:

```
test_input/

└── patient_001_0000.nii.gz
```

nnUNet automatically performs:

* Image resampling
* Intensity normalization
* Foreground cropping
* Patch-based inference
* Restoration to original image geometry

Run inference:

```bash
python src/inference.py \
--input test_input \
--output ./output_mask \
--model ./models/model/nnUNetTrainerV2__nnUNetPlansv2.1
```

Generated segmentation:

```
output_mask/

└── patient_001.nii.gz
```

---

# Evaluation Performance

Performance was evaluated using Dice similarity coefficient (DSC).

| Structure | Dice Score      |
| --------- | --------------- |
| Liver     | 0.9718 ± 0.0191 |
| Tumor     | 0.7584 ± 0.2590 |

---

# Demo Visualization

Example segmentation generated using the provided inference pipeline.

The visualization includes:

* CT image slice
* Liver segmentation
* Tumor segmentation

Input CT:

![CT slice](demo/ct_slice.png)

Segmentation result:

![Segmentation result](demo/example_result.png)

---

# Containerized Deployment

This repository includes a container-based inference architecture designed for scalable deployment.

The deployment consists of two services:

```
                    User
                      |
                      |
              FastAPI REST API
                      |
             ------------------
             Shared Volume
             ------------------
                      |
          nnUNet Inference Service
                      |
                      |
          Liver + Tumor Segmentation
```

## API Service

The API layer provides:

* CT file upload
* Input validation
* Automatic nnUNet filename conversion
* Inference request handling
* Segmentation result delivery

## Inference Service

The nnUNet service provides:

* GPU-enabled inference
* Five-fold ensemble prediction
* Automatic preprocessing
* Segmentation generation

---

# Docker Deployment

## Requirements

* Docker
* Docker Compose
* NVIDIA GPU (recommended)
* NVIDIA Container Toolkit

Build containers:

```bash
docker compose build
```

Start services:

```bash
docker compose up
```

The inference API will be available at:

```
http://localhost:8000
```

---

# API Usage

Example request:

```bash
curl \
-X POST \
-F "file=@patient.nii.gz" \
http://localhost:8000/segment \
-o segmentation.nii.gz
```

The service automatically handles:

1. CT image upload
2. nnUNet filename conversion:

```
patient.nii.gz

        ↓

patient_0000.nii.gz
```

3. nnUNet preprocessing
4. Five-fold ensemble inference
5. Segmentation mask generation

---

# Dataset

The model was trained using a multi-institutional CT dataset of patients with colorectal liver metastases.

Due to data privacy restrictions, the training dataset is not publicly released.

---

# Intended Use

This repository is intended for:

* Research in medical image analysis
* Development of AI-assisted radiology tools
* Benchmarking segmentation algorithms
* Reproducible evaluation of nnUNet-based pipelines

## Limitations

* Performance may vary across CT scanners, acquisition protocols, contrast phases, and patient populations.
* The model was developed specifically for colorectal liver metastasis cases.
* External validation is required before clinical deployment.
* This model is not intended for autonomous clinical decision-making.

---

# Citation

If you use this model, please cite:

```bibtex
@article{hamghalam2026liver,
title={Liver cancer segmentator: Metadata-guided confidence scoring for reliable segmentation of colorectal liver metastases in CT},
author={Hamghalam, Mohammad and others},
journal={Computer Methods and Programs in Biomedicine},
volume={276},
pages={109233},
year={2026}
}
```

Please also cite nnUNet:

```bibtex
@article{isensee2021nnunet,
title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
author={Isensee, Fabian and others},
journal={Nature Methods},
year={2021}
}
```

---

# License

Please refer to the repository license and dataset restrictions before using this model.
