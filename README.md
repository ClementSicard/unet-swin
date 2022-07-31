# Road Segmentation Project - Computational Intelligence Lab (2022)

- [Road Segmentation Project - Computational Intelligence Lab (2022)](#road-segmentation-project---computational-intelligence-lab-2022)
	- [Set up the environnement](#set-up-the-environnement)
	- [Download data](#download-data)
	- [Run the code](#run-the-code)
		- [Run the baselines](#run-the-baselines)
			- [Support Vector Classifier baseline (`baseline-svc`)](#support-vector-classifier-baseline-baseline-svc)
			- [Patch-CNN baseline (`baseline-patch-cnn`)](#patch-cnn-baseline-baseline-patch-cnn)
			- [Vanilla-UNet baseline (`baseline-unet`)](#vanilla-unet-baseline-baseline-unet)
		- [Run our models](#run-our-models)
			- [Fine-tuning UNet (`unet`)](#fine-tuning-unet-unet)
			- [USwinBaseNet (`swin-unet`)](#uswinbasenet-swin-unet)
	- [Create an ensemble submission](#create-an-ensemble-submission)

## Set up the environnement

Install `conda`, then:

```bash
conda create -n cil python=3.8
conda activate cil
pip install -r requirements.txt
pre-commit install
```

## Download data

From the root of this repository:

```bash
kaggle competitions download -c cil-road-segmentation-2022
unzip cil-road-segmentation-2022.zip
mkdir data
mv training data
mv test data
```

Then with `python`:

```python
from glob import sample
import os

VAL_SIZE = 10

for img in sample(glob("data/training/images/*.png"), VAL_SIZE):
	os.rename(img, img.replace('training', 'validation'))
	mask = img.replace('images', 'groundtruth')
	os.rename(mask, mask.replace('training', 'validation'))

```

## Run the code

The `code/run.py` script runs with the following arguments, and **needs to be executed from root directoy of this repo**:

```bash
python code/run.py 	[-h] [--model-type {small,base}]
			[--loss {bce,dice,mixed,focal,twersky,f1,patch-f1}]
			--train-dir TRAIN_DIR [--model-save-dir MODEL_SAVE_DIR] [--no-augment]
			--val-dir VAL_DIR --test-dir TEST_DIR [--n_epochs N_EPOCHS]
			[--batch_size BATCH_SIZE] [--checkpoint_path CHECKPOINT_PATH]
			{baseline-svc,baseline-unet,baseline-patch-cnn,unet,swin-unet}
```

| Argument               | Description                                                                                                |                                  Choices                                   |    Default value     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------: | :------------------: |
| `model` (_positional_) | Model to use for training                                                                                  | `baseline-svc`, `baseline-unet`, `baseline-patch-cnn`, `unet`, `swin-unet` |          -           |
| `--train-dir`          | Path to the training directory                                                                             |                                     -                                      | `None`, **required** |
| `--val-dir`            | Path to the validation directory                                                                           |                                     -                                      | `None`, **required** |
| `--test-dir`           | Path to the validation directory                                                                           |                                     -                                      | `None`, **required** |
| `--model-save-dir`     | Path where the model will be saved                                                                         |                                     -                                      |         `.`          |
| `--model-save-dir`     | Path where the model will be saved                                                                         |                                     -                                      |         `.`          |
| `--no-augment`         | If added to the command, the dataset will not be augmented (i.e. the initial dataset will be used instead) |                                     -                                      |         `.`          |
| `--n_epochs`           | Number of epochs to train on                                                                               |                                     -                                      |        `100`         |
| `--batch_size`         | Path of a model checkpoint to load to resume training                                                      |                                     -                                      |        `128`         |
| `--model-type`         | For model `swin-unet` (will be ignored otherwise), to select which pre-trained model we would like to use  |                              `small`, `base`                               |        `base`        |
| `--checkpoint_path`    | Path of a model checkpoint to load to resume training                                                      |                                     -                                      |        `None`        |
| `-l`, `--loss`         | Loss to train with                                                                                         |        `bce`, `dice`, `mixed`, `focal`, `twersky`, `f1`, `patch-f1`        |        `bce`         |

### Run the baselines

From the **root of this folder**:

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
SAVE_DIR="<where you want the best weights to be stored>"

python code/run.py baseline-unet \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
  --n_epochs 35 \
  --batch_size 4 \
  --model-save-dir $SAVE_DIR
```

### Run our models

#### Fine-tuning UNet ([`unet`](code/models/swin_unet.py))

Basic usage:

```bash
SAVE_DIR="<where you want the best weights to be stored>"
N_EPOCHS=200
BATCH_SIZE=4
LOSS="<choose your loss>"

python code/run.py unet \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
  --loss $LOSS \
  --n_epochs $N_EPOCHS \
  --batch_size $BATCH_SIZE \
  --model-save-dir $SAVE_DIR
```

#### USwinBaseNet ([`swin-unet`](code/models/swin_unet.py))

[Link to the paper](Paper%20report.pdf)

Basic usage:

```bash
SAVE_DIR="<where you want the best weights to be stored>"
N_EPOCHS=200
BATCH_SIZE=2
LOSS="<choose your loss>"

python code/run.py swin-unet \
  --train-dir "data/training" \
  --test-dir "data/test" \
  --val-dir "data/validation" \
  --loss $LOSS \
  --n_epochs $N_EPOCHS \
  --batch_size $BATCH_SIZE \
  --model-save-dir $SAVE_DIR
```

## Create an ensemble submission

You can make create an ensemble submission with this Python script, with which you can add as many `.csv` files as you want.

The final result is determined by majority vote.

```bash
python tools/create_ensemble_submission.py file1.csv file2.csv ... -o "submission/ensemble.csv"
```
