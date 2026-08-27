# MedSegOps

[![CI](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/ci.yml/badge.svg)](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/ci.yml)
[![CD](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/cd.yml/badge.svg)](https://github.com/Harish-Kurla-Shankarareddy/MedSegOps/actions/workflows/cd.yml)

**Production-oriented medical image segmentation and MLOps pipeline for 3D spleen segmentation using MONAI, PyTorch, FastAPI, DICOM/DICOM SEG, Explainable AI, Docker, CI/CD, and automated model-quality validation.**

---

## Overview

MedSegOps is an end-to-end medical AI application for 3D spleen segmentation from CT images.

The project combines a MONAI/PyTorch segmentation model with the engineering required to build, evaluate, explain, test, secure, and deliver the application as a deployable service.

The system supports NIfTI and DICOM CT input, segmentation statistics, DICOM SEG export, Explainable AI, automated model-quality checks, Docker deployment, and GitHub Actions CI/CD.

> **Research and engineering project:** MedSegOps is not intended for clinical diagnosis, treatment, or direct patient care.

---

## Key Features

- 3D spleen segmentation with a MONAI UNet
- NIfTI (`.nii`, `.nii.gz`) input
- DICOM CT series input
- Binary DICOM SEG export
- Spleen volume and voxel statistics
- 3D Grad-CAM
- Occlusion sensitivity
- XAI alignment evaluation
- Dice and IoU evaluation
- Automated model regression quality gate
- FastAPI web interface and API
- Dockerized deployment
- Non-root container execution
- Docker health checks
- Trivy vulnerability scanning
- Git LFS model-weight management
- GitHub Actions CI/CD
- GitHub Container Registry delivery

---

## Architecture

```text
                    Medical CT Data
                          │
                 ┌────────┴────────┐
                 │                 │
               NIfTI             DICOM
                 │                 │
                 └────────┬────────┘
                          │
                          ▼
                  Preprocessing
                          │
                          ▼
                 MONAI 3D UNet
                          │
                          ▼
                  Spleen Segmentation
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      Statistics          XAI          DICOM SEG
                          │
                  ┌───────┴───────┐
                  │               │
              Grad-CAM        Occlusion
                  │               │
                  └───────┬───────┘
                          ▼
                   XAI Evaluation
                          │
                          ▼
                     FastAPI
                          │
                          ▼
                       Docker
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
               CI                   CD
                │                   │
        Tests / Quality /       GHCR publish
        Docker / Trivy