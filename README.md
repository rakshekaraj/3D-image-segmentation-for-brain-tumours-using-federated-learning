# 3D Image Segmentation for Brain Tumours using Federated Learning

This repository contains our implementation of federated learning for 3D brain tumor segmentation using the BraTS2020 dataset. The goal was to explore privacy-preserving deep learning techniques that still perform well under real-world constraints like data heterogeneity.

We implemented standard centralized training, vanilla federated learning (FedAvg), and personalization via FedPer. We also began developing an idea called the Contribution Factor, aimed at improving how client updates are aggregated in federated setups.

## Project Highlights

- Built a 3D U-Net model for multi-modal MRI segmentation.
- Simulated federated learning across 5 clients with both IID and non-IID splits.
- Evaluated personalization using FedPer, where each client keeps a local head.
- Introduced the concept of Contribution Factor (early-stage, not yet formalized).

## Dataset

We used the BraTS2020 dataset, which includes:

- Four MRI modalities: T1, T1CE, T2, and FLAIR
- Labels for whole tumor, tumor core, and enhancing tumor
- Preprocessing: 128×128×128 resizing, intensity normalization, and mask binarization

## Model Architecture

The segmentation model is based on 3D U-Net, which extends the 2D U-Net to work with volumetric data. It uses an encoder-decoder structure with skip connections to preserve spatial detail while learning contextual features. The model takes 4-channel MRI inputs and produces a 3-class segmentation output.

Loss function: Dice Loss  
Optimizer: Adam

## Code Structure

- `BRATS_RESUNet.ipynb`: Centralized baseline training
- `BraTS_Federated.ipynb`: Standard FL with FedAvg and equal splits
- `BraTS_Federated_Non-IID.ipynb`: FL with unequal client data
- `BraTS_FedPer.ipynb`: FL with personalization (FedPer)
- `model.py`: Model definitions

Start with the centralized baseline, then move into the federated notebooks.

## Results Summary

| Method                   | Dice Score | IoU     |
|--------------------------|------------|---------|
| Centralized              | 0.708      | 0.680   |
| FedAvg (equal)           | 0.6941     | 0.5695  |
| FedAvg (non-IID)         | 0.4648     | 0.3195  |
| FedPer (non-IID)         | 0.4640     | 0.4653  |

## Contribution Factor

We're exploring a new way to assign weights to each client's update during aggregation. Instead of relying only on data size, the Contribution Factor considers how useful or stable each client's updates are. Details are still evolving and will be shared in future work.

## How to Run

1. Clone the repository  
2. Set up your Python environment (we recommend conda or virtualenv)  
3. Open the notebooks in your preferred Jupyter environment  
4. Follow the order: centralized → federated → personalized

```bash
git clone https://github.com/rakshekaraj/3D-image-segmentation-for-brain-tumours-using-federated-learning
cd 3D-image-segmentation-for-brain-tumours-using-federated-learning
