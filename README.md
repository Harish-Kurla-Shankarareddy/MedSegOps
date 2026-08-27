# MedSegOps

[![CI](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/ci.yml/badge.svg)](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/ci.yml)
[![CD](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/cd.yml/badge.svg)](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/cd.yml)

**Production-oriented medical image segmentation and MLOps platform for 3D spleen segmentation using MONAI, PyTorch, FastAPI, DICOM/DICOM SEG, Explainable AI, Docker, GitHub Actions, automated model-quality validation, and container security.**

---

## Overview

MedSegOps is an end-to-end medical AI and MLOps project for 3D spleen segmentation from CT images.

The project combines a MONAI/PyTorch segmentation model with the software engineering required to:

- process medical images
- perform 3D segmentation
- calculate quantitative segmentation statistics
- generate explainable-AI visualizations
- export DICOM SEG
- evaluate model quality using Dice and IoU
- detect model regressions
- containerize the application
- scan the container for security vulnerabilities
- automatically test the project
- publish the Docker image through CI/CD

The application provides a browser-based FastAPI interface as well as an API-based workflow.

> **Research and engineering project:** MedSegOps is not intended for clinical diagnosis, treatment, or direct patient care.

---

# Visual Demo

## 3D Spleen Segmentation

The MedSegOps interface displays the original CT volume, predicted spleen segmentation, segmentation overlay, and quantitative statistics.

![MedSegOps Segmentation Result](docs/segmentation-result.png)

## Explainable AI — 3D Grad-CAM

3D Grad-CAM highlights regions of the CT volume that contribute to the segmentation prediction.

![MedSegOps Grad-CAM](docs/xai-gradcam.png)

## Explainable AI — Occlusion Sensitivity

Occlusion sensitivity provides a complementary explanation by systematically masking local regions of the input volume and measuring changes in model output.

![MedSegOps Occlusion Sensitivity](docs/xai-occlusion.png)

## CI/CD Pipeline

GitHub Actions automatically tests, evaluates, builds, scans, and delivers the application.

![MedSegOps GitHub Actions](docs/github-actions.png)

---

# Key Features

## Medical AI

- 3D spleen segmentation
- MONAI 3D UNet
- PyTorch inference
- Sliding-window 3D inference
- Medical-image preprocessing
- Quantitative segmentation statistics

## Medical Imaging

- NIfTI input
- DICOM CT series input
- DICOM-to-volume reconstruction
- NIfTI segmentation output
- Binary DICOM SEG export
- DICOM SEG validation

## Explainable AI

- 3D Grad-CAM
- Occlusion sensitivity
- XAI threshold analysis
- Explanation precision
- Explanation coverage
- Explanation IoU

## Model Evaluation

- Dice Similarity Coefficient
- Intersection over Union (IoU)
- Multi-case model benchmark
- Minimum and mean metrics
- Automated model regression quality gate

## MLOps / Engineering

- FastAPI application
- Docker containerization
- Non-root container execution
- Docker health checks
- Trivy security scanning
- Git LFS model storage
- GitHub Actions CI
- GitHub Actions CD
- GitHub Container Registry (GHCR)

---

# Architecture

```text
                         Medical CT Data
                               |
                    +----------+----------+
                    |                     |
                  NIfTI                 DICOM
                    |                     |
                    +----------+----------+
                               |
                               v
                    Input Validation
                               |
                               v
                         Preprocessing
                               |
                               v
                      MONAI 3D UNet
                               |
                               v
                     Spleen Segmentation
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
         Statistics            XAI          DICOM SEG
                                |
                       +--------+--------+
                       |                 |
                   Grad-CAM         Occlusion
                       |                 |
                       +--------+--------+
                                |
                                v
                         XAI Evaluation
                                |
                                v
                            FastAPI
                                |
                                v
                             Docker
                                |
                     +----------+----------+
                     |                     |
                     v                     v
                    CI                     CD
                     |                     |
          Tests / Model Quality       GHCR Publish
          Docker / Trivy
```

---

# Medical AI Pipeline

MedSegOps uses a 3D MONAI UNet for volumetric spleen segmentation.

The inference pipeline performs:

1. Medical-image loading
2. Channel normalization
3. Orientation normalization
4. Spatial resampling
5. Intensity normalization
6. Sliding-window 3D inference
7. Post-processing
8. Inverse transformation
9. Binary spleen-mask generation
10. Output writing

## Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MONAI UNet |
| Spatial dimensions | 3D |
| Input channels | 1 |
| Output classes | 2 |
| Normalization | Batch normalization |
| Input spacing | 1.5 × 1.5 × 2.0 mm |
| Sliding-window ROI | 96 × 96 × 96 |
| Sliding-window batch size | 4 |
| Sliding-window overlap | 0.5 |

The segmentation model uses sliding-window inference to process full 3D volumes while maintaining local volumetric context.

---

# Supported Medical Images

## NIfTI

MedSegOps supports NIfTI medical volumes including:

```text
.nii
.nii.gz
```

The volume is loaded and passed through the MONAI preprocessing and inference pipeline.

## DICOM

DICOM CT studies can be uploaded as a series of individual slices.

The application reconstructs the DICOM series into a 3D volume before segmentation.

```text
DICOM CT Series
      |
      v
Series Reconstruction
      |
      v
3D Medical Volume
      |
      v
Preprocessing
      |
      v
MONAI 3D UNet
      |
      v
Spleen Segmentation
```

---

# Segmentation Output and Statistics

The inference pipeline produces a binary spleen segmentation mask.

The application also calculates quantitative information describing the segmentation.

## Reported Statistics

| Statistic | Description |
|---|---|
| Spleen voxel count | Number of voxels belonging to the predicted spleen |
| Spleen volume | Physical spleen volume calculated from voxel spacing |
| Image dimensions | Dimensions of the input 3D volume |
| Voxel spacing | Physical spacing between voxels |

## Example

```text
Input Type:        DICOM
Spleen Volume:     247.05 mL
Spleen Voxels:     51,811
Voxel Spacing:     0.9766 × 0.9766 × 5.0 mm
Image Dimensions:  512 × 512 × 55
```

The generated segmentation mask can be written as NIfTI and can also be exported as DICOM SEG.

---

# DICOM and DICOM SEG

MedSegOps supports DICOM CT input and binary DICOM SEG output.

## DICOM Processing Workflow

```text
DICOM CT Slices
       |
       v
Metadata / Ordering
       |
       v
Series Reconstruction
       |
       v
3D Volume
       |
       v
Segmentation
       |
       v
Binary Spleen Mask
       |
       v
DICOM SEG
```

## DICOM SEG

The application exports the predicted spleen mask as a binary DICOM Segmentation Object.

The generated SEG maintains the relationship between:

- source CT images
- segmentation mask
- segment metadata
- referenced image instances

## DICOM SEG Validation

Generated DICOM SEG objects can be validated using:

```bash
python scripts/validate_dicom_seg.py
```

Validation checks include:

- DICOM SEG SOP Class
- Modality
- Segmentation type
- Segment number
- Segment label
- Referenced source images
- Decoded segmentation shape
- Pixel values
- Segmentation voxel count

The project also includes scripts for DICOM and DICOM SEG testing:

```text
scripts/test_dicom.py
scripts/test_dicom_seg.py
scripts/create_test_dicom.py
scripts/validate_dicom_seg.py
```

---

# Explainable AI

MedSegOps provides two complementary explainability methods for the 3D segmentation model.

## 3D Grad-CAM

Grad-CAM is used to identify regions of the 3D CT volume that contribute to the segmentation prediction.

The project evaluates explanation alignment using:

- Precision
- Coverage
- Explanation IoU

### Observed Grad-CAM Evaluation

| Metric | Result |
|---|---:|
| Precision | **94.74%** |
| Coverage | **88.60%** |
| Explanation IoU | **84.45%** |

The best observed explanation threshold was:

```text
80%
```

![MedSegOps Grad-CAM](docs/xai-gradcam.png)

## Occlusion Sensitivity

Occlusion sensitivity provides a complementary explanation method.

The approach systematically masks local regions of the input volume and measures the resulting change in model output.

### Observed Occlusion Evaluation

| Metric | Result |
|---|---:|
| Precision | **69.65%** |
| Coverage | **70.30%** |
| Explanation IoU | **53.81%** |

![MedSegOps Occlusion Sensitivity](docs/xai-occlusion.png)

### XAI Evaluation Scripts

The repository contains dedicated evaluation scripts including:

```text
scripts/evaluate_xai.py
scripts/evaluate_xai_thresholds.py
scripts/evaluate_decoder_xai.py
scripts/evaluate_occlusion_xai.py
scripts/visualize_gradcam.py
scripts/visualize_decoder_gradcam.py
scripts/visualize_occlusion.py
```

> Explanation metrics measure the quality/alignment of the explanation and are distinct from segmentation metrics such as Dice and IoU.

---

# Model Quality

MedSegOps evaluates segmentation quality using quantitative overlap metrics.

The project implements:

- Dice Similarity Coefficient
- Intersection over Union (IoU)

## Five-Case Regression Benchmark

The current fixed engineering regression benchmark uses:

```text
spleen_10
spleen_12
spleen_13
spleen_14
spleen_16
```

## Benchmark Results

| Metric | Result |
|---|---:|
| Mean Dice | **0.9802** |
| Mean IoU | **0.9611** |
| Minimum Dice | **0.9754** |
| Minimum IoU | **0.9519** |

## Per-Case Results

| Case | Dice | IoU |
|---|---:|---:|
| spleen_10 | 0.9772 | 0.9553 |
| spleen_12 | 0.9838 | 0.9682 |
| spleen_13 | 0.9830 | 0.9666 |
| spleen_14 | 0.9815 | 0.9636 |
| spleen_16 | 0.9754 | 0.9519 |

## Regression Quality Gate

The CI quality gate currently requires:

| Gate | Threshold |
|---|---:|
| Mean Dice | ≥ 0.95 |
| Mean IoU | ≥ 0.92 |
| Minimum Dice | ≥ 0.93 |
| Minimum IoU | ≥ 0.88 |

Current model:

```text
Mean Dice: 0.9802
Mean IoU : 0.9611
Min Dice : 0.9754
Min IoU  : 0.9519

QUALITY GATE: PASS
```

Run the benchmark locally:

```bash
python scripts/evaluate_model_quality.py
```

Run the regression quality gate:

```bash
python scripts/check_model_quality.py
```

> These are engineering regression thresholds intended to detect model degradation. They are not clinical acceptance thresholds and do not constitute clinical validation.

---

# Web Application

MedSegOps provides a browser-based interface through FastAPI.

The web application supports:

- Medical image upload
- NIfTI workflows
- DICOM workflows
- Segmentation inference
- Segmentation visualization
- Spleen statistics
- Explainable AI
- NIfTI output
- DICOM SEG output

## Run Locally

Create and activate the Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn app.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Health Endpoint

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "MedSegOps"
}
```

---

# Docker

MedSegOps is packaged as a Docker image containing:

- Python runtime
- Medical-AI dependencies
- FastAPI application
- MONAI model
- Model weights
- Runtime libraries

## Build

```bash
docker build -t medsegops:hardened .
```

## Run

```bash
docker run -d \
  --name medsegops-hardened \
  -p 8000:8000 \
  medsegops:hardened
```

Open:

```text
http://127.0.0.1:8000
```

## Container Status

```bash
docker ps
```

## Logs

```bash
docker logs medsegops-hardened
```

## Container Health

The image includes a Docker `HEALTHCHECK` that verifies the FastAPI `/health` endpoint.

```bash
docker inspect \
  --format='{{.State.Health.Status}}' \
  medsegops-hardened
```

Expected:

```text
healthy
```

## Non-Root Execution

The application runs as the dedicated non-root user:

```text
medsegops
```

Example:

```text
uid=999(medsegops)
gid=999(medsegops)
```

---

# Docker Compose

The repository also contains:

```text
docker-compose.yml
```

Compose can be used to manage the containerized application configuration.

Start the Compose application from the repository directory with:

```bash
docker compose up --build
```

Stop it with:

```bash
docker compose down
```

> Docker Compose is included to provide a reproducible container configuration and a path toward multi-service deployment.

---

# Container Security

MedSegOps includes multiple container-security measures.

## Non-Root Container

The application does not run as the Docker `root` user.

## Operating-System Security Updates

The Docker image applies Debian package upgrades during the build process.

## Minimal Runtime Dependencies

Only required runtime operating-system packages are installed.

## Health Check

The container includes a Docker health check for:

```text
/health
```

## Trivy Vulnerability Scanning

The Docker image is scanned for HIGH and CRITICAL OS/library vulnerabilities.

Current hardened-image result:

```text
HIGH:     0
CRITICAL: 0
```

The security scan is integrated into CI.

---

# Git LFS and Model Artifacts

The trained segmentation model is stored using Git Large File Storage (Git LFS).

Model artifact:

```text
models/monai/model.pt
```

Git LFS is used because model weights are binary artifacts and should not be managed like ordinary source-code files.

## Verify Git LFS

```bash
git lfs version
```

## Retrieve Model Weights

```bash
git lfs pull
```

## Verify Model

```bash
ls -lh models/monai/model.pt
```

The CI pipeline verifies that the model artifact exists and is an actual binary file rather than an LFS pointer.

---

# CI/CD Pipeline

MedSegOps uses GitHub Actions for continuous integration and continuous delivery.

## Continuous Integration

The CI workflow performs:

```text
Checkout
   |
Git LFS
   |
Model Artifact Verification
   |
Install Dependencies
   |
pytest
   |
Model Quality Gate
   |
Docker Build
   |
API Smoke Test
   |
Trivy Security Scan
```

### Software Tests

The project currently has:

```text
16 passed
```

### Model Quality

CI evaluates the fixed five-case benchmark and checks:

```text
Mean Dice >= 0.95
Mean IoU  >= 0.92
Min Dice  >= 0.93
Min IoU   >= 0.88
```

### Docker Validation

CI also verifies:

- Docker image builds
- container starts
- FastAPI health endpoint responds
- Trivy finds no HIGH/CRITICAL vulnerabilities

## Continuous Delivery

The CD workflow builds and publishes the Docker image to:

**GitHub Container Registry (GHCR)**

Images are tagged with:

```text
latest
commit SHA
```

Example:

```text
ghcr.io/<github-owner>/medsegops:latest
```

and:

```text
ghcr.io/<github-owner>/medsegops:sha-<commit>
```

## Current CI/CD Status

```text
MedSegOps CI   ✅
MedSegOps CD   ✅
```

---

# Model Validation Data

The model-quality benchmark uses a fixed set of five Spleen dataset cases.

The validation data is treated as an external dataset and is not committed to the repository.

The CI workflow prepares the required validation data before executing the model-quality benchmark.

The repository contains:

```text
scripts/download_ci_validation_data.py
```

which prepares the validation cases used by:

```text
scripts/check_model_quality.py
```

The downloaded validation archive and extracted medical images are kept outside Git tracking.

---

# Installation

## Requirements

Recommended environment:

```text
Python 3.12
Git
Git LFS
Docker
```

## Clone Repository

```bash
git clone https://github.com/Harish-Kurla-Shankarareddy/MedSegOps.git
cd MedSegOps
```

## Initialize Git LFS

```bash
git lfs install
git lfs pull
```

## Create Virtual Environment

Linux / WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Running Locally

## Python / FastAPI

```bash
uvicorn app.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Docker

```bash
docker build -t medsegops:hardened .
```

```bash
docker run -d \
  --name medsegops-hardened \
  -p 8000:8000 \
  medsegops:hardened
```

Open:

```text
http://127.0.0.1:8000
```

---

# Testing

MedSegOps contains unit, integration, model, XAI, DICOM, and API-related tests.

## Run All Tests

```bash
pytest -q
```

Current result:

```text
16 passed
```

## Model Quality Benchmark

```bash
python scripts/evaluate_model_quality.py
```

## Model Quality Gate

```bash
python scripts/check_model_quality.py
```

Expected:

```text
QUALITY GATE: PASS
```

## DICOM Testing

```bash
python scripts/test_dicom.py
```

## DICOM SEG Testing

```bash
python scripts/test_dicom_seg.py
```

## DICOM SEG Validation

```bash
python scripts/validate_dicom_seg.py
```

## XAI Testing

The repository includes dedicated scripts for Grad-CAM and occlusion sensitivity evaluation and visualization.

---

# Project Structure

```text
MedSegOps/
|
+-- app/
|   |
|   +-- api/
|   |   +-- main.py
|   |
|   +-- io/
|   |   +-- __init__.py
|   |   +-- dicom.py
|   |   +-- dicom_seg.py
|   |
|   +-- pipeline/
|   |   +-- inference.py
|   |
|   +-- xai/
|       +-- alignment.py
|       +-- gradcam.py
|       +-- occlusion.py
|
+-- evaluation/
|   +-- metrics.py
|
+-- models/
|   +-- monai/
|       +-- model.pt
|
+-- scripts/
|   +-- check_model_quality.py
|   +-- create_test_dicom.py
|   +-- download_ci_validation_data.py
|   +-- evaluate_model_quality.py
|   +-- evaluate_xai.py
|   +-- evaluate_xai_thresholds.py
|   +-- evaluate_decoder_xai.py
|   +-- evaluate_occlusion_xai.py
|   +-- test_dicom.py
|   +-- test_dicom_seg.py
|   +-- test_gradcam.py
|   +-- test_occlusion.py
|   +-- validate_dicom_seg.py
|   +-- visualize_gradcam.py
|   +-- visualize_decoder_gradcam.py
|   +-- visualize_occlusion.py
|
+-- tests/
|   |
|   +-- model/
|   |   +-- test_spleen_model.py
|   |
|   +-- integration/
|   |   +-- conftest.py
|   |   +-- test_api_health.py
|   |   +-- test_dicom_conversion.py
|   |   +-- test_dicom_seg.py
|   |
|   +-- unit/
|   |   +-- test_metrics.py
|   |
|   +-- xai/
|
+-- docs/
|   +-- segmentation-result.png
|   +-- xai-gradcam.png
|   +-- xai-occlusion.png
|   +-- github-actions.png
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml
|       +-- cd.yml
|
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- pyproject.toml
+-- .dockerignore
+-- .gitattributes
+-- .gitignore
+-- README.md
```

---

# Application Workflow

```text
                       User
                        |
                        v
                Upload Medical Image
                        |
              +---------+---------+
              |                   |
              v                   v
            NIfTI               DICOM
              |                   |
              |              Series Reconstruction
              |                   |
              +---------+---------+
                        |
                        v
                  Preprocessing
                        |
                        v
                 MONAI 3D UNet
                        |
                        v
                  Segmentation
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
      Statistics        XAI       DICOM SEG
                        |
                  +-----+-----+
                  |           |
              Grad-CAM    Occlusion
                  |           |
                  +-----+-----+
                        |
                        v
                 FastAPI Response
```

---

# MLOps Workflow

```text
                    Developer
                        |
                        v
                     Git Push
                        |
                        v
                GitHub Actions CI
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
        pytest     Model Quality   Docker
                        |             |
                        |             v
                        |        Smoke Test
                        |             |
                        |             v
                        |          Trivy
                        |             |
          +-------------+-------------+
                        |
                        v
                    CI PASS
                        |
                        v
                Continuous Delivery
                        |
                        v
                     GHCR
                        |
                        v
                 Deployable Image
```

> The final repository should keep the CI and CD workflows consistent with the actual workflow dependency configuration.

---

# Reproducibility

MedSegOps separates different classes of project artifacts:

```text
Source code
Model weights
External medical validation data
Generated outputs
```

## Source Code

Version controlled with Git.

## Model Weights

Stored through Git LFS:

```text
models/monai/model.pt
```

## External Medical Data

The Spleen validation dataset is treated as an external dependency and is not committed to the repository.

The CI workflow prepares the required benchmark data before evaluation.

## Generated Outputs

Generated segmentation outputs and runtime data are excluded from source control.

This separation reduces repository size and prevents generated or external medical data from becoming part of normal source-code history.

---

# Engineering Quality Gates

MedSegOps uses multiple independent validation layers.

```text
                 MedSegOps Quality
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
   Software Tests   Model Quality   Container Security
        |               |               |
      pytest         Dice / IoU         Trivy
        |               |               |
        +---------------+---------------+
                        |
                        v
                   Docker Smoke Test
                        |
                        v
                   CI / CD Pipeline
```

This separates:

- software correctness
- model-performance regression
- container security
- runtime health

rather than treating them as one check.

---

# Performance Baseline

Current five-case engineering benchmark:

```text
Mean Dice: 0.9802
Mean IoU : 0.9611
Min Dice : 0.9754
Min IoU  : 0.9519
```

The model passed the configured regression thresholds for all four values.

These measurements represent the current fixed regression benchmark and should not be interpreted as independent clinical test-set performance.

---

# Limitations

MedSegOps is a research and engineering prototype.

The current reported model-performance results are based on a fixed five-case regression benchmark.

The benchmark is designed to detect model degradation during development and CI/CD.

The benchmark should not be interpreted as:

- Clinical validation
- Regulatory validation
- Clinical acceptance criteria
- A clinical performance guarantee
- A replacement for expert annotation
- A clinically approved diagnostic system

The model has not been established as suitable for diagnosis or treatment decisions.

Broader independent evaluation across larger, diverse, and clinically representative datasets would be required before any clinical application.

---

# Future Work

Potential future improvements include:

## Model Validation

- Independent external test-set evaluation
- Larger validation cohorts
- Cross-dataset evaluation
- Model calibration
- Robustness testing

## Uncertainty and Reliability

- Predictive uncertainty estimation
- Out-of-distribution detection
- Failure-case analysis
- Confidence calibration

## MLOps

- Experiment tracking
- Model registry
- Model version promotion
- Automated model deployment
- Model monitoring
- Data-drift monitoring
- Performance monitoring

## Deployment

- GPU-optimized inference
- Triton Inference Server
- Kubernetes
- Cloud deployment
- Horizontal scaling
- Observability and metrics

## Medical Interoperability

- Broader DICOM interoperability testing
- Additional DICOM object types
- More comprehensive metadata validation
- Integration testing with PACS-style workflows

---

# Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch |
| Medical AI | MONAI |
| Segmentation | 3D UNet |
| Medical Imaging | NIfTI, DICOM |
| DICOM SEG | highdicom, pydicom |
| Image Processing | NumPy, nibabel, SimpleITK |
| Explainability | Grad-CAM, Occlusion Sensitivity |
| API | FastAPI |
| ASGI Server | Uvicorn |
| Testing | pytest |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Security | Trivy |
| Version Control | Git |
| Model Artifact Storage | Git LFS |
| CI/CD | GitHub Actions |
| Container Registry | GitHub Container Registry |

---

# Repository Badges

The repository exposes GitHub Actions status for:

```text
CI
CD
```

These badges at the top of the README provide a quick indication of the current automated pipeline status.

---

# Quick Reference

## Start FastAPI

```bash
uvicorn app.api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

## Build Docker Image

```bash
docker build -t medsegops:hardened .
```

## Run Docker

```bash
docker run -d \
  --name medsegops-hardened \
  -p 8000:8000 \
  medsegops:hardened
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Run Tests

```bash
pytest -q
```

## Run Model Benchmark

```bash
python scripts/evaluate_model_quality.py
```

## Run Model Quality Gate

```bash
python scripts/check_model_quality.py
```

## Validate DICOM SEG

```bash
python scripts/validate_dicom_seg.py
```

## Docker Security Scan

```bash
docker run --rm \
  -v trivy-cache:/root/.cache/trivy \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest \
  image \
  --scanners vuln \
  --severity HIGH,CRITICAL \
  --ignore-unfixed \
  --skip-java-db-update \
  --skip-files "**/pip/_vendor/bom.cdx.json" \
  medsegops:hardened
```

---

# Disclaimer

**MedSegOps is a research and engineering project and is not intended for clinical diagnosis, treatment, or direct patient care.**

The segmentation model and software pipeline have not been clinically validated or approved for medical decision-making.

The model-quality measurements reported in this repository are engineering benchmarks used to evaluate software/model regression and do not constitute clinical validation, regulatory approval, clinical safety evidence, or evidence of effectiveness in patient care.

---

# Author

**Harish Kurla Shankarareddy**

Medical AI · Computer Vision · Deep Learning · Medical Image Analysis · MLOps

GitHub:

https://github.com/Harish-Kurla-Shankarareddy/MedSegOps