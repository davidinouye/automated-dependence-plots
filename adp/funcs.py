import numpy as np
import pandas as pd


def is_categorical(dtypes):
    def check_dtype(dtype):
        if pd.api.types.is_categorical_dtype(dtype):
            return True
        try:
            dtype.categories
        except AttributeError:
            try:
                dtype['categories']
            except TypeError:
                return False     
            except KeyError:
                return False
            else:
                return True
        else:
            return True
    return np.array([check_dtype(dtype) for dtype in dtypes])

def get_dtype_categories(dtype):
    try:
        return dtype.categories
    except AttributeError:
        return dtype['categories']

def img2vec(img):
    if img.ndim == 4:
        n, c, w, h = img.shape
        return img.reshape(n, c * w * h)
    c, w, h = img.shape
    return img.reshape(c * w * h)

# NOTE values are hardcoded
def vec2img(v):
    if v.ndim == 2:
        n, _ = v.shape
        return v.reshape(n, 3, 32, 32)
    return v.reshape(3, 32, 32)
