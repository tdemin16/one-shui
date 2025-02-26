

<h1 align="center">
Unlearning Personal Data from a Single Image</br>(TMLR 2025)
</h1>

<div align="center">

#### [Thomas De Min](https://scholar.google.com/citations?user=fnh_i0cAAAAJ&hl=en), [Massimiliano Mancini](https://scholar.google.com/citations?hl=it&authuser=1&user=bqTPA8kAAAAJ), [Stéphane Lathuilière](https://scholar.google.fr/citations?user=xllguWMAAAAJ&hl=fr), </br>[Subhankar Roy](https://scholar.google.it/citations?user=YfzgrDYAAAAJ&hl=en), and [Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ&hl=it&authuser=1)

[![Paper](https://img.shields.io/badge/Paper-arxiv.2407.12069-B31B1B.svg)](https://arxiv.org/abs/2407.12069)
</div>

> **Abstract.** 
*Machine unlearning aims to erase data from a model as if the latter never saw them during training. While existing approaches unlearn information from complete or partial access to the training data, this access can be limited over time due to privacy regulations. Currently, no setting or benchmark exists to probe the effectiveness of unlearning methods in such scenarios. To fill this gap, we propose a novel task we call One-Shot Unlearning of Personal Identities (1-SHUI) that evaluates unlearning models when the training data is not available. We focus on unlearning identity data, which is specifically relevant due to current regulations requiring personal data deletion after training. To cope with data absence, we expect users to provide a portraiting picture to aid unlearning. We design requests on CelebA, CelebA-HQ, and MUFAC with different unlearning set sizes to evaluate applicable methods in 1-SHUI. Moreover, we propose MetaUnlearn, an effective method that meta-learns to forget identities from a single image. Our findings indicate that existing approaches struggle when data availability is limited, especially when there is a dissimilarity between the provided samples and the training data.*

## Citation
TODO

## Installation 
### Dependencies
We used Python 3.11.7 for all our experiments. Therefore, we suggest creating a conda environment as follows:
```bash
$ conda create -n oneshui python=3.11.7
```
and install pip requirements with:
```bash
pip install -r requirements.txt
```

### Pre-trained weights
All ViT pre-trained weights are automatically downloaded using timm.

### Datasets
Use `--data_dir path/to/datasets/` when running our scripts to set the location of your dataset.
Then, our code will look for datasets inside the provided directory, expecting each dataset to be structured as follows:
```
mufac/
    ├── test_images/
    ├── fixed_val_dataset/
    ├── val_images/
    ├── train_images/
    ├── fixed_test_dataset/
    ├── custom_val_dataset.csv
    ├── custom_train_dataset.csv
    └── custom_test_dataset.csv
celeba/
    ├── img_align_celeba/
    ├── list_bbox_celeba.txt
    ├── list_attr_celeba.txt
    ├── list_eval_partition.csv
    ├── identity_CelebA.txt
    ├── list_attr_celeba.csv
    ├── list_eval_partition.txt
    └── list_landmarks_align_celeba.txt
CelebAMask-HQ/
    ├── CelebA-HQ-img/
    ├── CelebAMask-HQ-attribute-anno.txt
    └── CelebA-HQ-identity.txt
```

> Note: We got CelebA-HQ identity annotations from [here](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/tree/main)

## Run Experiments
Run each experiment using `python main.py` with different arguments. Run `python main.py --help` for the full list of available arguements. Below we report a couple of scenarios.

### Pretrain
Model pre-training on CelebA for 5 identities:
```bash
$ python main.py --method pretrain --dataset celeba --num_identities 5
```

### Retrain
Model retraining on CelebA-HQ for 20 identities:
```bash
$ python main.py --method retrain --dataset celebahq --num_identities 20
```

### MetaUnlearn
Train and eval MetaUnlearn on Mufac with 10 identities:
```bash
$ python main.py --method 
