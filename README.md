```markdown
# Liver Cancer Segmentator (LCS)

**Liver Cancer Segmentator: Metadata-Guided Confidence Scoring for Reliable Segmentation of Colorectal Liver Metastases in CT**

A deep learning framework for automatic and reliable segmentation of liver parenchyma and colorectal liver metastases (CRLM) from contrast-enhanced abdominal CT images.

This repository provides:

- A reproducible nnUNet-based 3D segmentation pipeline
- Automatic model downloading from Hugging Face
- Five-fold ensemble inference
- Multi-class liver and tumor segmentation
- Container-ready deployment architecture for AI-assisted radiology workflows

The trained model checkpoints are publicly available:

**Hugging Face Model Repository**

https://huggingface.co/Hamghalam/liver-cancer-segmentation-nnunet


---

# Overview

Accurate segmentation of colorectal liver metastases (CRLM) on contrast-enhanced CT is essential for quantitative assessment, treatment planning, and AI-assisted radiology applications.

This project introduces the **Liver Cancer Segmentator (LCS)**, a deep learning model designed for robust segmentation of liver parenchyma and metastatic tumors from abdominal CT examinations.

The model was trained and evaluated using a multi-institutional dataset containing:

- **446 contrast-enhanced CT examinations**
- **355 cases for training**
- **91 cases for testing**
- Data from multiple clinical cohorts and treatment settings

The model performs multi-class segmentation:

| Label | Structure |
|------|-----------|
| 0 | Background |
| 1 | Liver parenchyma |
| 2 | Colorectal liver metastasis tumor |


---

# End-to-End LCS Pipeline

The complete workflow combines automated segmentation with confidence-based quality assessment.

The pipeline includes:

1. CT image preprocessing
2. nnUNet-based segmentation
3. Confidence estimation
4. Automatic acceptance or rejection of segmentation results
5. Radiologist review for low-confidence cases


![LCS pipeline](figures/fig1_pipeline.png)


**Figure 1. End-to-end LCS pipeline with confidence-based quality control.**  
Input CT scans are preprocessed and segmented using nnUNet. The generated segmentation is evaluated using confidence scoring. High-confidence predictions are accepted automatically, while low-confidence predictions are flagged for expert review.


---

# Multi-Institutional Dataset

The model was developed using CT examinations collected from diverse clinical cohorts representing different disease stages and treatment conditions.

The dataset includes:

- Chemotherapy cohort
- Resection cohort
- TCIA cohort
- All stages-MDA cohort
- All stages-MSK cohort


![Dataset cohorts](figures/fig2_cohorts.png)


**Figure 2. Representative contrast-enhanced CT samples from five cohorts used for model development.**  
The examples demonstrate the variability of CRLM cases, including differences in disease burden, tumor appearance, and imaging characteristics across institutions.


---

# Model Architecture

The segmentation model is based on:

- **Framework:** nnUNet v1
- **Architecture:** 3D U-Net
- **Configuration:** 3D Full Resolution
- **Training strategy:** Five-fold cross-validation ensemble
- **Deep learning framework:** PyTorch


During inference, predictions from all five trained folds are combined:

```

```
             CT Image

                |
      nnUNet preprocessing

    -------------------------
    |    |    |    |        |
  fold0 fold1 fold2 fold3 fold4

    -------------------------

                |
      Ensemble prediction

                |
      Liver + Tumor mask
```

````


---

# Installation

## Clone repository

```bash
git clone https://github.com/hamghalam/liverCancerSegmentator.git

cd liverCancerSegmentator
````

## Create environment

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

The trained nnUNet checkpoints are available through Hugging Face.

Download:

```bash
python scripts/download_model.py
```

Model structure:

```
models/

└── model/
    └── nnUNetTrainerV2__nnUNetPlansv2.1/

        ├── plans.pkl

        ├── fold_0/
        ├── fold_1/
        ├── fold_2/
        ├── fold_3/
        └── fold_4/
```

---

# Inference

Prepare input CT scans in NIfTI format:

```
test_input/

└── patient_001_0000.nii.gz
```

Run inference:

```bash
python src/inference.py \
--input test_input \
--output ./output_mask \
--model ./models/model/nnUNetTrainerV2__nnUNetPlansv2.1
```

The generated segmentation masks will be saved:

```
output_mask/

└── patient_001.nii.gz
```

The inference pipeline automatically performs:

* Image resampling
* Intensity normalization
* Foreground cropping
* Patch-based prediction
* Restoration to original image geometry

---

# Performance

Performance was evaluated on 91 independent test CT examinations.

| Structure | Dice Score                     |
| --------- | ------------------------------ |
| Liver     | 0.9707 (95% CI: 0.9663–0.9751) |
| Tumor     | 0.7695 (95% CI: 0.7166–0.8224) |

The model also demonstrated improved reliability through confidence scoring by incorporating:

* Tumor volume
* CT slice thickness

This reduced the area under the risk coverage curve from **16.7 to 10.3**, improving automatic failure detection.

---

# Demo Visualization

A demonstration example is provided using non-patient data.

The demo contains:

* Input CT slice
* Predicted liver segmentation
* Predicted tumor segmentation

```
demo/

├── ct_slice.png
└── example_result.png
```

---

# Deployment Architecture

The repository supports containerized deployment for integration into clinical AI workflows.

Architecture:

```
              User

               |
               |

        FastAPI REST API

               |

        ----------------

        Shared Volume

        ----------------

               |

       nnUNet Inference Service

               |

        Liver + Tumor Mask
```

## API Container

Responsibilities:

* CT file upload
* Input validation
* Automatic nnUNet filename handling
* Request management
* Result delivery

## Segmentation Container

Responsibilities:

* GPU-based inference
* Five-fold ensemble prediction
* nnUNet preprocessing
* Segmentation generation

Run with Docker:

```bash
docker compose build

docker compose up
```

---

# Intended Use

This project is intended for:

* Medical image analysis research
* AI-assisted radiology development
* Benchmarking liver and tumor segmentation algorithms
* Reproducible nnUNet-based segmentation experiments

---

# Limitations

* Performance may vary across scanners, acquisition protocols, contrast phases, and patient populations.
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

Please refer to the repository license and dataset restrictions before use.


