from PIL import Image
import glob
import numpy as np


def load_dataset(dirpath: str):
    return np.array([Image.open(f) for f in glob.glob(f"{dirpath}/*")])
