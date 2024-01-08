import torch
from torch import nn


class MultiHeadModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(MultiHeadModel, self).__init__()
        self.input_dim = in_channels
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