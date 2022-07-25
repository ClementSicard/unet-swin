# Road Segmentation Project - Computational Intelligence Lab (2022)

- [Road Segmentation Project - Computational Intelligence Lab (2022)](#road-segmentation-project---computational-intelligence-lab-2022)
  - [Set up l'environnement](#set-up-lenvironnement)
  - [Load le dataset augmenté](#load-le-dataset-augmenté)
  - [Sources](#sources)
  - [Dataset](#dataset)
    - [Augmenter le dataset](#augmenter-le-dataset)
  - [Liens utiles](#liens-utiles)

## Set up l'environnement

Installer `conda`, puis:

```bash
conda create -n cil python=3.8
conda activate cil
pip install -r requirements.txt
pre-commit install
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
