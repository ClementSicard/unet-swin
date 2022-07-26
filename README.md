# Road Segmentation Project - Computational Intelligence Lab (2022)

- [Road Segmentation Project - Computational Intelligence Lab (2022)](#road-segmentation-project---computational-intelligence-lab-2022)
  - [Set up the environnement](#set-up-the-environnement)
  - [Run the baselines](#run-the-baselines)
    - [Our runs to obtain the scores in the paper](#our-runs-to-obtain-the-scores-in-the-paper)
      - [Support Vector Classifier baseline (`baseline-svc`)](#support-vector-classifier-baseline-baseline-svc)
      - [Patch-CNN baseline (`baseline-patch-cnn`)](#patch-cnn-baseline-baseline-patch-cnn)
      - [Vanilla-UNet baseline (`baseline-unet`)](#vanilla-unet-baseline-baseline-unet)
  - [Load le dataset augmenté](#load-le-dataset-augmenté)
  - [Sources](#sources)
  - [Dataset](#dataset)
    - [Augmenter le dataset](#augmenter-le-dataset)
  - [Liens utiles](#liens-utiles)

## Set up the environnement

Install `conda`, then:

```bash
conda create -n cil python=3.8
conda activate cil
pip install -r requirements.txt
pre-commit install
```

## Run the baselines

From the **root of this folder**:

```bash
python code/run.py "name of the baseline" \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
  --n_epochs 20
```

with the names in

```python
{"baseline-svc", "baseline-unet", "baseline-patch-cnn"}
```

### Our runs to obtain the scores in the paper

#### Support Vector Classifier baseline ([`baseline-svc`](code/models/baselines/baseline_svm_classifier.py))

```bash
python code/run.py baseline-svc \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
```

#### Patch-CNN baseline ([`baseline-patch-cnn`](code/models/baselines/baseline_patch_cnn.py))

```bash
python code/run.py baseline-patch-cnn \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
  --n_epochs 20 \
  --batch_size 128
```

#### Vanilla-UNet baseline ([`baseline-unet`](code/models/baselines/baseline_vanilla_unet.py))

```bash
python code/run.py baseline-unet \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
  --n_epochs 35 \
  --batch_size 5 # Might be modified when ran on Euler
```

## Load le dataset augmenté

```python
from utils import *

BATCH_SIZE = 32
device = get_best_available_device()

dataset = ImageDataset(path="../data/training/", device=device, use_patches=False)
dataloader = iter(DataLoader(dataset, batch_size=32, shuffle=True))
```

Pour itérer sur les paires image/mask du `DataLoader`:

```python
batch = next(dataloader)

for image, mask in zip(batch[0], batch[1]):
    plt.imshow(image.to('cpu').numpy().transpose(1, 2, 0))
    plt.imshow(mask.to('cpu').numpy().transpose(1, 2, 0), alpha=0.3)
    plt.show()
```

## Sources

## Dataset

### Augmenter le dataset

- [Lien du tuto PyTorch](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms)

## Liens utiles

[Semester Project Infos](https://docs.google.com/document/d/1kXMPYBRJYzMQNceVUpsaQBSMec8kuaAHiIRGQuG9DQA/edit)
[Road Segmentation Infos](https://docs.google.com/document/d/1MVRFu4oKWgAluY7CRzehFH8Pt-TNSW_9JJ6E9gmraZg/edit#heading=h.go9uiolcl982)
[Kaggle Competition](https://www.kaggle.com/t/32523d9cf29948e089dc4c7a23ecf549)
