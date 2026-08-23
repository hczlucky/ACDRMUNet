# ACDRMUNet

This is the official PyTorch implementation of **ACDRMUNet: Anatomically Calibrated Deformable Reconstruction Mamba U-Net for Medical Image Segmentation**.

## Abstract

ACDRMUNet is a Vision Mamba segmentation network with anatomically explicit semantic calibration (AESC) and uncertainty-adaptive deformable reconstruction (UADR). AESC forms anatomy-aware semantic priors at multiple decoder stages, while UADR uses the predicted semantic probability and uncertainty to guide progressive feature reconstruction.

## 0. Main Environments

Create and activate the environment, then install the dependencies:

```bash
conda create -n acdrmunet python=3.10
conda activate acdrmunet
pip install -r requirements.txt
```

The selective-scan extension in `mamba-ssm` must be built for the installed PyTorch and CUDA versions.

## 1. Prepare the Datasets

### Synapse

The preprocessed Synapse dataset is available from the [CCViM Google Drive link](https://drive.google.com/file/d/1-eDXzTgXrTTo7hcrWZnh_wVEtB92PBNz/view?usp=sharing). This release follows the Synapse data format used by [zymissy/CCViM](https://github.com/zymissy/CCViM).

Place the dataset under `./data/Synapse/`:

```text
data/Synapse/
├── lists/
│   └── lists_Synapse/
│       ├── all.lst
│       ├── test_vol.txt
│       └── train.txt
├── test_vol_h5/
│   └── caseXXXX.npy.h5
└── train_npz/
    └── caseXXXX_sliceXXX.npz
```

### BUSI

Download BUSI from [Kaggle](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) or the [original dataset page](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset). Following [jeya-maria-jose/UNeXt-pytorch](https://github.com/jeya-maria-jose/UNeXt-pytorch/), the experiments use the 647 benign and malignant cases with a fixed split of 517 training and 130 test images. Normal cases are not used because they contain no lesion regions.

```text
inputs/BUSI/
├── images/
│   └── *.png
└── masks/
    └── 0/
        └── *.png
```

The current public training code, dataset loader, and configuration are provided only for Synapse. BUSI is documented here as the second dataset used in the paper; no BUSI implementation is included in this repository.

## 2. Configure the Paths

Edit the relative paths in `configs/config_synapse.py` if your dataset or pretrained checkpoint is stored elsewhere. The default locations are:

```text
data/Synapse/
pre_trained_weights/vmamba_small_e238_ema.pth
```

## 3. Train ACDRMUNet

Run training from the repository root:

```bash
python train_synapse.py
```

To resume a run:

```bash
python train_synapse.py --resume results/<run_name>/latest.pth
```

Training outputs are saved under `results/`.

## 4. Main Files

```text
configs/config_synapse.py          Synapse training configuration
datasets/dataset_synapse.py        Synapse data loading and augmentation
models/backbone.py                 L-VSS backbone components
models/acdrmunet/                  ACDRMUNet, AESC, and UADR
losses/acdrmunet_loss.py           Segmentation and auxiliary losses
train_synapse.py                   Training entry
engine.py                          Training and volume-wise validation
```

## 5. Citation

The paper is currently under submission. Citation information will be updated after publication.

## 6. Acknowledgments

This repository is built upon and inspired by [MFEVM-UNet](https://github.com/loveAI666/MFEVM-UNet), [VM-UNet](https://github.com/JCruan519/VM-UNet), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet). We thank the authors for their public implementations.
