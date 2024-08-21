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

def compute_windowed_rmse(lta, window_size = 10000):
    """ Calclute the RMSE deviation over the window size for the lta series"""
    # convert lta to a numpy array if not already
    if not isinstance(lta, np.ndarray):
        lta = np.array(lta)
    rmse = np.zeros(len(lta))
    for i in range(len(lta)):
        if i < window_size:
            rmse[i] = np.sqrt(np.mean((lta[:i] - lta[:i].mean())**2))
        else:
            rmse[i] = np.sqrt(np.mean((lta[i-window_size:i] - lta[i-window_size:i].mean())**2))
    return rmse
