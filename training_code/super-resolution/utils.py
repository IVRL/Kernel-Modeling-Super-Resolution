import numpy as np
import pywt

def to_np(tensor):
    tensor = tensor.detach().cpu().numpy()
    tensor = np.squeeze(tensor)
    if tensor.ndim == 3:
        tensor = np.rollaxis(tensor, 0, 3)
    return tensor

def get_model_dir():
    return 'checkpoint/'
