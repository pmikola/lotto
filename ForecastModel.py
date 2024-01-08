import random
import torch
from torch import nn


class ForecastModel(torch.nn.Module):
    def __init__(self, in_channels, no_bits):
        super(ForecastModel, self).__init__()
        self.input_dim = in_channels * no_bits
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
        # print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)

        return a, b, c, d, e, f  # , g  # G IS FOR EUROJACKPOT