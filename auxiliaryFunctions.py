import subprocess
import sys
from itertools import chain

import numpy as np
import torch


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))


def create_dataset(dataset, lookback):
    X = torch.zeros((dataset[:].shape[0], lookback, dataset[:].shape[1]))
    y = torch.zeros((dataset[:].shape[0], lookback, dataset[:].shape[1]))

    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X[i] = feature
        y[i] = target
    return X, y


def test_model(test_input, model, device, path, dataset):
    model.load_state_dict(torch.load(path + model.__class__.__name__ + '.pth'))
    model.train(mode=False)
    model = model.to(device)
    test_input = torch.unsqueeze(
        torch.unsqueeze(torch.tensor(np.array(test_input, dtype=float), dtype=torch.float32), 0), 0).to(device)

    prediction = model(test_input)
    preds = tuple(t.cpu() for t in prediction)
    PREDS = []

    for i in range(0, len(preds)):
        PREDS.append(preds[i][0].item())

    rounded_preds = [round(x, 2) for x in PREDS]
    print(rounded_preds, '\nShould be ', dataset[-1])


def findLeastFreqElementMapping(arr):
    dictMap = {}
    for i in range(len(arr)):
        if (arr[i] in dictMap.keys()):
            dictMap[arr[i]] += 1
        else:
            dictMap[arr[i]] = 1
    leastElementCtr = min(dictMap.values())
    for i in dictMap:
        if dictMap[i] == leastElementCtr:
            leastElement = i
            break
    return leastElement, leastElementCtr


def noise_psd(N, psd=lambda f: 1):
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    # Normalize S
    S = S / np.sqrt(np.mean(S ** 2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def PSDGenerator(f):
    return lambda N: noise_psd(N, f)


@PSDGenerator
def white_noise(f):
    return 1


@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)


@PSDGenerator
def violet_noise(f):
    return f


@PSDGenerator
def brownian_noise(f):
    return 1 / np.where(f == 0, float('inf'), f)


@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))
