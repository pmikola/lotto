import csv
import random
import subprocess
import sys
# Lotto most used numbers
import sysconfig
import time
from collections import Counter
from heapq import nsmallest
from itertools import chain
from itertools import islice, groupby
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

from LottoFun.FNO import FNO2d
from LottoFun.ForecastModel import ForecastModel
from LottoFun.MultiHeadModel import MultiHeadModel
from LottoFun.trainingLoop import trainingLoop
from auxiliaryFunctions import *

# pip -V # where pip will install
# pip list -v
print(sysconfig.get_paths()["purelib"])  # where python look for packages
sys.path.append('C:/Python311/Lib/site-packages')
# sys.path[:0] = ['C:\Python311\Lib\site-packages\distfit']
from distfit import distfit

path = 'C:\\PYTHON_PROJECTS\\Lotto_fun\\LottoFun\\'
csv_file_path1 = 'szybkie600-wynikilottonetpl.csv'
csv_file_path2 = 'ekstra_pensja.csv'
csv_file_path3 = 'eurojackpot.csv'
csv_file_path4 = 'lotto.csv'

print("GPU ENABLED ? : ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

f = open(csv_file_path3)
rows = len(list(f))
lenum = 7
lookback = 2
print(rows)
LottoWins = np.empty([rows, lenum], 'int')
LottoWins_sorted = np.empty([rows, lenum], 'int')
separator = '\n-----------------------------------------------------\n'

with open(csv_file_path3, newline='\n') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # print(row[0:6])

        LottoWins[line_count, :] = row[2:]
        # print(LottoWins[line_count, :])
        line_count += 1
    print(f'Processed {line_count} lines.')

# print(LottoWins.shape[0])
d1_LottoWins = np.empty([LottoWins.shape[0] * LottoWins.shape[1]], 'int')
# print(d1_LottoWins.shape)
counter = 0
for i in range(0, LottoWins.shape[0]):
    d1_LottoWins[counter:counter + lenum] = LottoWins[i, :]
    counter += lenum

# MOST FREQUENT NUMBER
# -----------------------------------------------------
freq = np.bincount(d1_LottoWins)
most = np.argmax(freq)
print(separator, 'MOST FREQUENT NUMBER\n', most, separator, )

# MOST FREQUENT NUMBERS (6)
# -----------------------------------------------------
c = Counter(d1_LottoWins)
n = lenum - 1
# group the items having same count together
grouped = (list(group) for _, group in groupby(c.most_common(), key=itemgetter(1)))
# slice first `n` of them
top_n = islice(grouped, n)
# flatten
result = list(chain.from_iterable(top_n))
print(separator, 'MOST FREQUENT NUMBERS (6)\n', result, separator)

# LEAST FREQUENT NUMBER
# -----------------------------------------------------

leastFreqElementMapping, ctr3 = findLeastFreqElementMapping(d1_LottoWins)
print(separator, 'LEAST FREQUENT NUMBER\n', leastFreqElementMapping, separator)

# LEAST FREQUENT NUMBERS
# -----------------------------------------------------
least_frequent = []
count = Counter(d1_LottoWins)
# print(count)
x = list(count.keys())
y = list(count.values())

smallest = nsmallest(lenum, y)
for k in range(0, lenum):
    least_frequent.append(x[y.index(smallest[k])])

print(separator, 'LEAST FREQUENT NUMBERS\n', least_frequent, '\n', nsmallest(6, y), separator)
# NUMBERS HISTOGRAM
# -----------------------------------------------------
values = [key for key, value in count.items() if value == min(count.values())]
# plt.title('Number histogram frequencies')
# plt.bar(x, y)
# plt.grid()
# plt.show()


# NUMBERS HISTOGRAM LOG
# -----------------------------------------------------
# fig, ax = plt.subplots()
# plt.title('Number LOG histogram frequencies')
# ax.bar(x, y)
# ax.grid()
# ax.set_yscale("log")
# plt.show()

# HOW OFTEN NUMBER FREQ COMPARE
# -----------------------------------------------------
print(y)
ratio = min(y) / max(y)

print(separator, 'RATIO BETWEEN MOST AND LEAST COMMON\n', ratio, separator)

# MOST FREQUENT COMBINATION
# -----------------------------------------------------
for i in range(0, rows):
    LottoWins_sorted[i, :] = sorted(LottoWins[i, :])

with open(csv_file_path1, 'r') as file:
    csv_reader = csv.reader(file)
    data_list1 = []
    for row in csv_reader:
        data_list1.append(row)

with open(csv_file_path2, 'r') as file:
    csv_reader = csv.reader(file)
    data_list2 = []
    for row in csv_reader:
        data_list2.append(row)

with open(csv_file_path3, 'r') as file:
    csv_reader = csv.reader(file)
    data_list3 = []
    for row in csv_reader:
        data_list3.append(row)

with open(csv_file_path4, 'r') as file:
    csv_reader = csv.reader(file)
    data_list4 = []
    for row in csv_reader:
        data_list4.append(row)

dataset1 = []
for row in data_list1:
    szybkie600 = row[0].split()[3:]
    dataset1.append([int(i) for i in szybkie600])

dataset2 = []
for row in data_list2:
    ekstraPensja = row[2:]
    dataset2.append([int(i) for i in ekstraPensja])

dataset3 = []
for row in data_list3:
    euroJackpot = row[2:]
    dataset3.append([int(i) for i in euroJackpot])

dataset4 = []
for row in data_list3:
    euroJackpot = row[2:]
    dataset3.append([int(i) for i in euroJackpot])

dataset = dataset2
tdataset = torch.Tensor(np.array(dataset))

############## What distributions have these datasetes? #############
def distfit(dataset,plot_flag):
    flattened_dataset_main = [item[0:5] for item in dataset]
    flattened_dataset_adNumb = [item[-1] for item in dataset]

    # Initialize model
    dfitMain = distfit()
    dfitAdNumb = distfit()

    # Find best theoretical distribution for empirical data X
    dfitMain.fit_transform(np.array(flattened_dataset_main))
    # Find best theoretical distribution for empirical data X
    dfitAdNumb.fit_transform(np.array(flattened_dataset_adNumb))
    if plot_flag == 1:
        fig, ax = dfitMain.plot(chart='pdf', n_top=11)
        fig, ay = dfitMain.plot(chart='cdf', n_top=11)
        fig, az = dfitMain.qqplot(np.array(flattened_dataset_main), n_top=11)
        plt.show()
        fig, ax = dfitAdNumb.plot(chart='pdf', n_top=11)
        fig, ay = dfitAdNumb.plot(chart='cdf', n_top=11)
        fig, az = dfitAdNumb.qqplot(np.array(flattened_dataset_adNumb), n_top=11)
        # plt.show()
    # Xnew = dfit.generate(5)





# Xnew = dfit.generate(5)


train_size = int(0.8 * len(tdataset))
test_size = int(0.15 * len(tdataset)) + 1
val_size = int(0.05 * len(tdataset))
print(train_size + test_size + val_size, len(dataset))
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(tdataset, [train_size, test_size, val_size])

train_datasetX, train_datasetY = create_dataset(train_dataset, lookback)
test_datasetX, test_datasetY = create_dataset(test_dataset, lookback)
val_datasetX, val_datasetY = create_dataset(val_dataset, lookback)

fig, ax = plt.subplots()
maximum = torch.amax(tdataset)
plot2dTimeSeries = False
if plot2dTimeSeries is True:
    pc = ax.pcolormesh(tdataset, vmin=0, vmax=maximum)
    plt.colorbar(pc)
    ax.set_frame_on(False)  # remove all spines
    # plt.plot(datasetY)
    plt.show(block=False)
    plt.pause(25)
    plt.close()
else:
    pass

nonLinModel = MultiHeadModel(6)
loss_multihead = nn.L1Loss()
optimiser = optim.Adam(nonLinModel.parameters(), lr=1e-3)

trainingLoop(PATH=path,
             print_model=False,
             do_training=False,
             device=device,
             batch_size=32,
             lookback=lookback,
             n_epochs=1000,
             optimiser=optimiser,
             model=nonLinModel,
             loss_fn=loss_multihead,
             X_train=train_datasetX,
             X_test=test_datasetX,
             X_val=val_datasetX,
             Y_train=train_datasetY,
             Y_test=test_datasetY,
             Y_val=val_datasetY)

forecastModel = ForecastModel(in_channels=6, no_bits=7)
loss_forecast = nn.MSELoss(reduction='mean')
optimiser = optim.Adam(forecastModel.parameters(), lr=1e-3, amsgrad=True, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=1e-4)

trainingLoop(PATH=path,
             print_model=False,
             do_training=False,
             device=device,
             batch_size=36,
             lookback=lookback,
             n_epochs=10000,
             optimiser=optimiser,
             model=forecastModel,
             loss_fn=loss_forecast,
             X_train=train_datasetX,
             X_test=test_datasetX,
             X_val=val_datasetX,
             Y_train=train_datasetY,
             Y_test=test_datasetY,
             Y_val=val_datasetY)

batch_size = 32
FNO = FNO2d(modes1=16, modes2=16, hidden_width=64, batch_size=batch_size, lookback=lookback)
loss_fft = nn.MSELoss()
optimiser = optim.Adam(FNO.parameters(), lr=1e-3, amsgrad=False, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=1e-4)

trainingLoop(PATH=path,
             print_model=False,
             do_training=True,
             device=device,
             batch_size=batch_size,
             lookback=lookback,
             n_epochs=20000,
             optimiser=optimiser,
             model=FNO,
             loss_fn=loss_fft,
             X_train=train_datasetX,
             X_test=test_datasetX,
             X_val=val_datasetX,
             Y_train=train_datasetY,
             Y_test=test_datasetY,
             Y_val=val_datasetY)

test_input = tdataset[-lookback - 1:-1, :]
print(test_input)
test_model(test_input, FNO, device, path, dataset)
