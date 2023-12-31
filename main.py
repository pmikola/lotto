import csv
import random
import time
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import csv
from collections import Counter
from itertools import chain, islice, groupby
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from heapq import nsmallest


# Lotto most used numbers



file = 'duzy_duzy+.txt'
f = open(file)
rows = len(list(f))
lenum = 6
print(rows)
LottoWins = np.empty([rows, lenum], 'int')
LottoWins_sorted = np.empty([rows, lenum], 'int')
separator = '\n-----------------------------------------------------\n'


with open(file, newline='\n') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        splited = row[0].split(' ')
        row[0] = splited[-1]
        # print(row[0:6])
        LottoWins[line_count, :] = row[0:lenum]
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
print(separator,'MOST FREQUENT NUMBER\n', most,separator, )

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
print(separator, 'MOST FREQUENT NUMBERS (6)\n', result,separator, )


# LEAST FREQUENT NUMBER
# -----------------------------------------------------
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


leastFreqElementMapping, ctr3 = findLeastFreqElementMapping(d1_LottoWins)
print(separator, 'LEAST FREQUENT NUMBER\n',leastFreqElementMapping,separator)

# LEAST FREQUENT NUMBERS
# -----------------------------------------------------
least_frequent = []
count = Counter(d1_LottoWins)
#print(count)
x = list(count.keys())
y = list(count.values())

smallest = nsmallest(lenum, y)
for k in range(0,lenum):
    least_frequent.append(x[y.index(smallest[k])])

print(separator,'LEAST FREQUENT NUMBERS\n',least_frequent,'\n',nsmallest(6, y),separator)
# NUMBERS HISTOGRAM
# -----------------------------------------------------
values = [key for key, value in count.items() if value == min(count.values())]

plt.bar(x, y)
plt.grid()
plt.show()


# NUMBERS HISTOGRAM LOG
# -----------------------------------------------------
fig, ax = plt.subplots()
ax.bar(x, y)
ax.grid()
ax.set_yscale("log")
plt.show()

# HOW OFTEN NUMBER FREQ COMPARE
# -----------------------------------------------------
print(y)
ratio = min(y)/max(y)

print(separator,'RATIO BETWEEN MOST AND LEAST COMMON\n',ratio,separator)

# MOST FREQUENT COMBINATION
# -----------------------------------------------------
for i in range(0,rows):
    LottoWins_sorted[i,:] = sorted(LottoWins[i,:])

# ho = []
#
# for i in range(0,rows):
#     for j in range(0,rows):
#         if i == j:
#             pass
#         else:
#             state = np.array_equal(LottoWins_sorted[i, :], LottoWins_sorted[j, :])
#             #print(state)
#             if state == False:
#                 pass
#             else:
#                 ho.append(j)
#                 ho.append(i)
#                 print(ho)




path = 'C:\\PYTHON_PROJECTS\\Lotto_fun\\LottoFun\\'
csv_file_path1 = 'szybkie600-wynikilottonetpl.csv'
csv_file_path2 = 'ekstra_pensja.csv'
csv_file_path3 = 'eurojackpot.csv'
print("GPU ENABLED ? : ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def create_dataset(dataset, lookback):
    X = torch.zeros((dataset[:].shape[0], lookback, dataset[:].shape[1]))
    y = torch.zeros((dataset[:].shape[0], lookback, dataset[:].shape[1]))

    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X[i] = feature
        y[i] = target
    return X, y


dataset = dataset2
tdataset = torch.Tensor(np.array(dataset))

train_size = int(0.8 * len(tdataset))
test_size = int(0.15 * len(tdataset)) + 1
val_size = int(0.05 * len(tdataset)) + 1
print(train_size + test_size + val_size, len(dataset))
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(tdataset, [train_size, test_size, val_size])

lookback = 2
train_datasetX, train_datasetY = create_dataset(train_dataset, lookback)
test_datasetX, test_datasetY = create_dataset(test_dataset, lookback)
val_datasetX, val_datasetY = create_dataset(val_dataset, lookback)

fig, ax = plt.subplots()
max = torch.amax(tdataset)
plot2dTimeSeries = False
if plot2dTimeSeries is True:
    pc = ax.pcolormesh(tdataset, vmin=0, vmax=max)
    plt.colorbar(pc)
    ax.set_frame_on(False)  # remove all spines
    # plt.plot(datasetY)
    plt.show(block=False)
    plt.pause(25)
    plt.close()
else:
    pass


def trainingLoop(PATH, print_model, do_training, device, batch_size, lookback, n_epochs, optimiser, model, loss_fn,
                 X_train,
                 X_test,
                 X_val, Y_train,
                 Y_test, Y_val):
    if print_model == True:
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    else:
        pass
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)
    model = model.to(device)
    model.hidden = (torch.randn(lookback, batch_size, model.input_dim).to(device),
                    torch.randn(lookback, batch_size, model.input_dim).to(device))
    best_loss = 100000.
    training_loss = []
    test_loss = []
    if do_training:
        for epoch in range(1, n_epochs + 1):
            optimiser.zero_grad()
            model.train(mode=True)
            loss_train = 0.
            loss_test = 0.
            indexes_train = random.sample(range(1, len(Y_train)), batch_size)
            indexes_test = random.sample(range(1, len(Y_test)), batch_size)
            output_train = model(X_train[indexes_train, :, :])
            output_test = model(X_test[indexes_test, :, :])

            for i in range(0, len(output_train)):
                for j in range(0, lookback):
                    # uSING RMSE LOSS (sqrt before mse)
                    loss_train += loss_fn(output_train[i],
                                          torch.unsqueeze(Y_train[indexes_train, j, i], dim=1))  # calculate loss
                    loss_test += loss_fn(output_test[i], torch.unsqueeze(Y_test[indexes_test, j, i], dim=1))

                    # print(output_train[i].shape,torch.unsqueeze(Y_train[indexes,i],dim=1).shape)
                    # time.sleep(0.5)

            loss_train = loss_train / len(output_train) / lookback
            loss_test = loss_test / len(output_test) / lookback

            loss_train.backward()
            optimiser.step()
            training_loss.append(loss_train.item())
            test_loss.append(loss_test.item())
            if epoch == 1 or epoch % 100 == 0:
                print(model.ampE, model.kk_percent)
                if loss_test < best_loss:
                    torch.save(model.state_dict(), PATH + model.__class__.__name__ + '.pth')
                    best_loss = loss_train
                print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                      f" Test loss {loss_test.item():.4f}")
            if epoch % 2500 == 0:

                model.load_state_dict(torch.load(PATH + model.__class__.__name__ + '.pth'))
                model.train(mode=False)
                indexes_val = random.sample(range(1, len(Y_val)), batch_size)
                loss_val = 0.
                output_val = model(X_val[indexes_val])
                for i in range(0, len(output_train)):
                    for j in range(0, lookback):
                        loss_val += loss_fn(output_val[i], torch.unsqueeze(Y_val[indexes_val, j, i], dim=1))
                loss_val = loss_val / len(output_val) / lookback
                print(f"----\nEpoch {epoch}, Validation loss {loss_val.item():.4f}\n----")
        plt.plot(training_loss, label='training loss')
        plt.plot(test_loss, label='test loss')
        # plt.yscale('log')
        plt.grid()
        plt.legend(loc="best")
        plt.show()
    else:
        print("Training not performed for : ", model.__class__.__name__)
    print("Done!")


class MultiHeadModel(torch.nn.Module):
    def __init__(self):
        super(MultiHeadModel, self).__init__()
        self.input_dim = 6
        self.h_dim1 = 512
        self.h_dim2 = 256
        self.h_dim3 = 128
        self.output_dim = 1
        self.p_drop = 0.1
        self.amplification = torch.nn.Parameter(torch.tensor(1.))
        self.amplification.requires_grad = True
        self.ampA = torch.nn.Parameter(torch.tensor(1.))
        self.ampA.requires_grad = True
        self.ampB = torch.nn.Parameter(torch.tensor(1.))
        self.ampB.requires_grad = True
        self.ampC = torch.nn.Parameter(torch.tensor(1.))
        self.ampC.requires_grad = True
        self.ampD = torch.nn.Parameter(torch.tensor(1.))
        self.ampD.requires_grad = True
        self.ampE = torch.nn.Parameter(torch.tensor(1.))
        self.ampE.requires_grad = True
        self.ampF = torch.nn.Parameter(torch.tensor(1.))
        self.ampF.requires_grad = True

        self.lin1 = nn.Linear(self.input_dim, self.h_dim1)
        self.lin2 = nn.Linear(self.h_dim1, self.h_dim2)
        self.AstroMask2 = nn.Linear(self.h_dim1, self.h_dim2)
        self.AstroMaskA = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskB = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskC = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskD = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskE = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskF = nn.Linear(self.h_dim2, self.h_dim3)
        self.linA = nn.Linear(self.h_dim2, self.h_dim3)
        self.linB = nn.Linear(self.h_dim2, self.h_dim3)
        self.linC = nn.Linear(self.h_dim2, self.h_dim3)
        self.linD = nn.Linear(self.h_dim2, self.h_dim3)
        self.linE = nn.Linear(self.h_dim2, self.h_dim3)
        self.linF = nn.Linear(self.h_dim2, self.h_dim3)
        self.head1 = nn.Linear(self.h_dim3, self.output_dim)
        self.head2 = nn.Linear(self.h_dim3, self.output_dim)
        self.head3 = nn.Linear(self.h_dim3, self.output_dim)
        self.head4 = nn.Linear(self.h_dim3, self.output_dim)
        self.head5 = nn.Linear(self.h_dim3, self.output_dim)
        self.head6 = nn.Linear(self.h_dim3, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform(self.weight)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        am2 = torch.sigmoid(self.AstroMask2(x))
        x = torch.tanh(self.lin2(x)) * torch.tanh(am2)
        astrocyteA = torch.sigmoid(self.AstroMaskA(x)) * self.amplification * torch.tanh(self.ampA)
        astrocyteB = torch.sigmoid(self.AstroMaskB(x)) * self.amplification * torch.tanh(self.ampB)
        astrocyteC = torch.sigmoid(self.AstroMaskC(x)) * self.amplification * torch.tanh(self.ampC)
        astrocyteD = torch.sigmoid(self.AstroMaskD(x)) * self.amplification * torch.tanh(self.ampD)
        astrocyteE = torch.sigmoid(self.AstroMaskE(x)) * self.amplification * torch.tanh(self.ampE)
        astrocyteF = torch.sigmoid(self.AstroMaskF(x)) * self.amplification * torch.tanh(self.ampF)
        A = torch.tanh(torch.nn.functional.dropout(self.linA(x), p=self.p_drop)) * astrocyteA
        B = torch.tanh(torch.nn.functional.dropout(self.linB(x), p=self.p_drop)) * astrocyteB
        C = torch.tanh(torch.nn.functional.dropout(self.linC(x), p=self.p_drop)) * astrocyteC
        D = torch.tanh(torch.nn.functional.dropout(self.linD(x), p=self.p_drop)) * astrocyteD
        E = torch.tanh(torch.nn.functional.dropout(self.linE(x), p=self.p_drop)) * astrocyteE
        F = torch.tanh(torch.nn.functional.dropout(self.linF(x), p=self.p_drop)) * astrocyteF
        a = self.head1(A) * torch.relu(self.ampA)
        b = self.head2(B) * torch.relu(self.ampB)
        c = self.head3(C) * torch.relu(self.ampC)
        d = self.head4(D) * torch.relu(self.ampD)
        e = self.head5(E) * torch.relu(self.ampE)
        f = self.head6(F) * torch.relu(self.ampF)

        return a, b, c, d, e, f


nonLinModel = MultiHeadModel()
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


class ForecastModel(torch.nn.Module):
    def __init__(self):
        super(ForecastModel, self).__init__()
        self.input_dim = 6 * 7
        self.h_dim1 = 512
        self.h_dim2 = 256
        self.h_dim3 = 128
        self.kk_percent = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentA = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentB = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentC = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentD = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentE = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentF = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.k_percentG = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.output_dim = 1
        self.p_drop = 0.1
        self.amplification = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.ampA = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.ampB = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.ampC = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.ampD = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.ampE = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        self.ampF = torch.nn.Parameter(torch.tensor(random.uniform(0., 1.)), requires_grad=True)
        # self.ampG = torch.nn.Parameter(torch.tensor(random.uniform(0., 10.)))
        # self.ampG.requires_grad = True

        self.lin1 = nn.Linear(self.input_dim, self.h_dim1)
        self.lstm = nn.LSTM(input_size=self.h_dim1, hidden_size=self.h_dim1, num_layers=2, batch_first=True)

        self.lin2 = nn.Linear(self.h_dim1, self.h_dim2)

        self.AstroMask2 = nn.Linear(self.h_dim1, self.h_dim2)

        self.AstroMaskA = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskB = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskC = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskD = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskE = nn.Linear(self.h_dim2, self.h_dim3)
        self.AstroMaskF = nn.Linear(self.h_dim2, self.h_dim3)
        # self.AstroMaskG = nn.Linear(self.h_dim2, self.h_dim3)
        self.linA = nn.Linear(self.h_dim2, self.h_dim3)
        self.linB = nn.Linear(self.h_dim2, self.h_dim3)
        self.linC = nn.Linear(self.h_dim2, self.h_dim3)
        self.linD = nn.Linear(self.h_dim2, self.h_dim3)
        self.linE = nn.Linear(self.h_dim2, self.h_dim3)
        self.linF = nn.Linear(self.h_dim2, self.h_dim3)
        # self.linG = nn.Linear(self.h_dim2, self.h_dim3)
        self.head1 = nn.Linear(self.h_dim3, self.output_dim)
        self.head2 = nn.Linear(self.h_dim3, self.output_dim)
        self.head3 = nn.Linear(self.h_dim3, self.output_dim)
        self.head4 = nn.Linear(self.h_dim3, self.output_dim)
        self.head5 = nn.Linear(self.h_dim3, self.output_dim)
        self.head6 = nn.Linear(self.h_dim3, self.output_dim)
        # self.head7 = nn.Linear(self.h_dim3, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform(self.weight)

    def binary(self, x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

    def mtopk(self, xs, k):
        _, indices = xs.topk(int(k), dim=-1)
        mask = torch.zeros_like(xs, dtype=torch.uint8)
        mask.scatter_(-1, indices, 1)
        return mask

    def forward(self, x):
        x_t = torch.zeros((x.size(dim=0), x.size(dim=1), x.size(dim=2), 7)).to(device)
        x_t[:, :, :, :] = self.binary(x[:, :, :].to(torch.int32), 7)[:]

        x = self.lin1(torch.flatten(x_t, start_dim=2))
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        am2 = torch.sigmoid(self.AstroMask2(x) * self.mtopk(self.lin2(x), self.kk_percent * self.h_dim2).to(torch.long))
        x = torch.tanh(self.lin2(x)) * am2
        top_kA = self.k_percentA * self.h_dim3
        top_kB = self.k_percentB * self.h_dim3
        top_kC = self.k_percentC * self.h_dim3
        top_kD = self.k_percentD * self.h_dim3
        top_kE = self.k_percentE * self.h_dim3
        top_kF = self.k_percentF * self.h_dim3
        # top_kG = int(self.k_percentG)

        top_k_astroA = self.mtopk(self.AstroMaskA(x), top_kA).to(torch.long)
        top_k_astroB = self.mtopk(self.AstroMaskB(x), top_kB).to(torch.long)
        top_k_astroC = self.mtopk(self.AstroMaskC(x), top_kC).to(torch.long)
        top_k_astroD = self.mtopk(self.AstroMaskD(x), top_kD).to(torch.long)
        top_k_astroE = self.mtopk(self.AstroMaskE(x), top_kE).to(torch.long)
        top_k_astroF = self.mtopk(self.AstroMaskF(x), top_kF).to(torch.long)
        # top_k_astroG = self.mtopk(self.AstroMaskG(x), top_kG).to(torch.long)

        astrocyteA = torch.sigmoid(self.AstroMaskA(x) * top_k_astroA) * self.amplification * self.ampA * 100
        astrocyteB = torch.sigmoid(self.AstroMaskB(x) * top_k_astroB) * self.amplification * self.ampB * 100
        astrocyteC = torch.sigmoid(self.AstroMaskC(x) * top_k_astroC) * self.amplification * self.ampC * 100
        astrocyteD = torch.sigmoid(self.AstroMaskD(x) * top_k_astroD) * self.amplification * self.ampD * 100
        astrocyteE = torch.sigmoid(self.AstroMaskE(x) * top_k_astroE) * self.amplification * self.ampE * 100
        astrocyteF = torch.sigmoid(self.AstroMaskF(x) * top_k_astroF) * self.amplification * self.ampF * 100
        # astrocyteG = torch.sigmoid(self.AstroMaskG(x)*top_k_astroG) * self.amplification * self.ampG

        A = torch.tanh(torch.nn.functional.dropout(self.linA(x), p=self.p_drop)) * astrocyteA
        B = torch.tanh(torch.nn.functional.dropout(self.linB(x), p=self.p_drop)) * astrocyteB
        C = torch.tanh(torch.nn.functional.dropout(self.linC(x), p=self.p_drop)) * astrocyteC
        D = torch.tanh(torch.nn.functional.dropout(self.linD(x), p=self.p_drop)) * astrocyteD
        E = torch.tanh(torch.nn.functional.dropout(self.linE(x), p=self.p_drop)) * astrocyteE
        F = torch.tanh(torch.nn.functional.dropout(self.linF(x), p=self.p_drop)) * astrocyteF
        # G = torch.tanh(torch.nn.functional.dropout(self.linG(x), p=self.p_drop)) * astrocyteG

        a = self.head1(A) * torch.relu(self.ampA * 10)
        b = self.head2(B) * torch.relu(self.ampB * 10)
        c = self.head3(C) * torch.relu(self.ampC * 10)
        d = self.head4(D) * torch.relu(self.ampD * 10)
        e = self.head5(E) * torch.relu(self.ampE * 10)
        f = self.head6(F) * torch.relu(self.ampF * 10)
        # g = self.head7(G) * torch.relu(self.ampG)  # g IS FOR EUROJACKPOT

        return a, b, c, d, e, f  # , g  # G IS FOR EUROJACKPOT


forecastModel = ForecastModel()
loss_forecast = nn.MSELoss(reduction='mean')
optimiser = optim.Adam(forecastModel.parameters(), lr=1e-3, amsgrad=True, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=1e-4)

trainingLoop(PATH=path,
             print_model=False,
             do_training=True,
             device=device,
             batch_size=64,
             lookback=lookback,
             n_epochs=7000,
             optimiser=optimiser,
             model=forecastModel,
             loss_fn=loss_forecast,
             X_train=train_datasetX,
             X_test=test_datasetX,
             X_val=val_datasetX,
             Y_train=train_datasetY,
             Y_test=test_datasetY,
             Y_val=val_datasetY)


def test_model(test_input, model, device, path):
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
    print(PREDS, '\nShould be : 16 18 23 24 27 + 01')


test_model([16., 28., 29., 30., 35., 04.], forecastModel, device, path)
