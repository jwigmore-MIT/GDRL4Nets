import numpy as np

def compute_lta(backlogs: np.ndarray):
    """
    Compute the long term average of the backlogs
    :param backlogs: backlog vector as a (N,) numpy array
    :return: long-term average backlogs as a (N,) numpy array
    """
    # compute cumalative sum of backlogs
    csum = np.cumsum(backlogs)
    divisor = np.arange(1, len(backlogs)+1)
    return np.divide(csum, divisor)