# ACDR-MUNet

This is the official PyTorch implementation of **ACDR-MUNet: Anatomically Calibrated Deformable Upsampling Mamba U-Net for Medical Image Segmentation**.

## Abstract

Accurate medical image segmentation requires not only effective contextual representation but also reliable spatial recovery of anatomical structures with substantial variations in shape, scale, and boundary characteristics. Although adaptive decoding strategies improve the flexibility of feature reconstruction, the spatial sampling process is generally driven by feature responses without sufficiently exploiting structured semantic information and its prediction reliability. To address this issue, we propose ACDR-MUNet, an anatomically calibrated deformable upsampling Mamba U-Net for medical image segmentation. At each decoding stage, the proposed Anatomical Equiangular Semantic Calibration (AESC) module establishes a structured semantic representation from decoder features and produces semantic evidence together with predictive uncertainty. These complementary cues are subsequently incorporated into an Uncertainty-Adaptive Deformable Upsampling (UADU) module, which dynamically regulates the effective sampling range and enables spatial reconstruction to adapt to different semantic structures and ambiguous regions. By coupling AESC and UADU throughout the decoder, ACDR-MUNet forms a progressive semantic-conditioned upsampling framework while preserving the efficient long-range modeling capability of Vision Mamba. Experiments over three independent runs on the Synapse multi-organ segmentation dataset and the BUSI breast ultrasound dataset demonstrate the effectiveness and generalization capability of the proposed framework. ACDR-MUNet achieves a mean Dice score of 84.12% and an HD95 of 13.36 mm on Synapse, together with a Dice score of 84.16%, an IoU of 75.48%, and an HD95 of 15.96 mm on BUSI. Component-wise ablation studies and qualitative comparisons further support the complementary contributions of semantic calibration and uncertainty-adaptive deformable upsampling.

## 0. Main Environments

Create and activate the environment, then install the dependencies:

```bash
conda create -n acdrmunet python=3.10
conda activate acdrmunet
pip install -r requirements.txt
```

## 1. Prepare the Datasets

### Synapse

Following [zymissy/CCViM](https://github.com/zymissy/CCViM), the Synapse dataset can be found here {[GoogleDrive](https://drive.google.com/file/d/1-eDXzTgXrTTo7hcrWZnh_wVEtB92PBNz/view?usp=sharing)}.

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

The BUSI dataset can be found [here](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset).

```text
inputs/BUSI/
├── images/
│   └── *.png
└── masks/
    └── 0/
        └── *.png
```

## 2. Prepare the Pre-trained Weights

The weights of the pre-trained VMamba can be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/13cTgCUhMTvuWQ8HxNVaB8g?pwd=iq3q). After that, the pre-trained weights should be stored in `./pretrained_weights/`.

## 3. Configure the Paths

Edit the relative paths in `configs/config_synapse.py` if your dataset or pretrained checkpoint is stored elsewhere. The default locations are:

```text
data/Synapse/
pretrained_weights/vmamba_small_e238_ema.pth
```

## 4. Train ACDR-MUNet

Run training from the repository root:

```bash
python train_synapse.py
```

To resume a run:

```bash
python train_synapse.py --resume results/<run_name>/latest.pth
```

Training outputs are saved under `results/`.

## 5. Main Files

```text
configs/config_synapse.py          Synapse training configuration
datasets/dataset_synapse.py        Synapse data loading and augmentation
models/backbone.py                 L-VSS backbone components
models/acdrmunet/                  ACDR-MUNet and decoder modules
losses/acdrmunet_loss.py           Segmentation and auxiliary losses
train_synapse.py                   Training entry
engine.py                          Training and volume-wise validation
```

## 6. Citation

The paper is currently under submission. Citation information will be updated after publication.

## 7. Acknowledgments

This repository is built upon and inspired by [MFEVM-UNet](https://github.com/loveAI666/MFEVM-UNet), [VM-UNet](https://github.com/JCruan519/VM-UNet), [VMamba](https://github.com/MzeroMiko/VMamba), and [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet). We thank the authors for their public implementations.
