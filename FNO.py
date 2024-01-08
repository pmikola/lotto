import math

import numpy as np
import torch
from torch import nn
import colorednoise as cn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, t1, t2, exp_r, exp_i, nAmp_real, nAmp_imag, nPhase_real, nphase_imag):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # print(t1.shape, t2.shape, '6')
        # Extract real and imaginary parts

        t1 = t1.unbind(dim=1)
        t2 = t2.unbind(dim=1)

        # for i in range(0,len(t1)):
        #     # print(t1[i].shape, t2[i].shape)
        #     step_dim.append(t1[i][:,None] * t2[i][None,:])
        # out = torch.stack(step_dim, dim=1)

        out = torch.stack([t1_i[:, None] * t2_i[None, :] for t1_i, t2_i in zip(t1, t2)], dim=1)
        out = out.sum(dim=2)
        d0 = out.size(dim=0)
        d1 = out.size(dim=1)
        d2 = out.size(dim=2)
        d3 = out.size(dim=3)

        e_r = torch.median(exp_r.flatten()).cpu().float().item()
        e_i = torch.median(exp_i.flatten()).cpu().float().item()
        amp_r = torch.median(nAmp_real.flatten()).cpu().float().item()
        amp_i = torch.median(nAmp_imag.flatten()).cpu().float().item()
        phi_r = torch.median(nPhase_real.flatten()).cpu().float().item()
        phi_i = torch.median(nphase_imag.flatten()).cpu().float().item()
        # print(int(phi_r), int(phi_i))
        noise_real = cn.powerlaw_psd_gaussian(e_r, d0 * d1 * d2 * d3)
        noise_real = np.roll(noise_real, int(phi_r))
        noise_real *= amp_r
        tnoise_real = torch.from_numpy(noise_real).to(device)
        reshaped_noise_real = torch.reshape(tnoise_real, (d0, d1, d2, d3))
        noise_imag = cn.powerlaw_psd_gaussian(e_i, d0 * d1 * d2 * d3)
        noise_imag = np.roll(noise_imag,int(phi_i))
        noise_imag *= amp_i
        tnoise_imag = torch.from_numpy(noise_imag).to(device)
        reshaped_noise_imag = torch.reshape(tnoise_imag, (d0, d1, d2, d3)) * amp_i
        noise_tensor = torch.complex(reshaped_noise_real, reshaped_noise_imag)
        out -= noise_tensor
        return out

    # return torch.einsum("bixy,ioxy->boxy", t1,t2)

    def forward(self, x, exp_r, exp_i, nAmp_real, nAmp_imag, nPhase_real, nphase_imag):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1, exp_r, exp_i, nAmp_real, nAmp_imag,
                             nPhase_real, nphase_imag)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2, exp_r, exp_i, nAmp_real, nAmp_imag,
                             nPhase_real, nphase_imag)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(torch.nn.Module):
    def __init__(self, modes1, modes2, hidden_width, batch_size, lookback):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.hidden_width = hidden_width
        self.batch_size = batch_size
        self.lookback = lookback
        self.padding = 9
        self.h_dim0 = 128
        self.h_dim1 = 64
        self.output_dim = 1
        # self.noiseExp = torch.nn.Parameter(torch.rand(1,  dtype=torch.cfloat))

        self.fc0 = torch.nn.Linear(6, self.hidden_width)
        # self.convT2d = torch.nn.ConvTranspose2d(self.hidden_width, self.hidden_width, kernel_size=2, stride=1)
        self.convS0 = SpectralConv2d(self.hidden_width, self.lookback, self.modes1, self.modes2)
        self.conv0 = nn.Conv2d(self.lookback, self.hidden_width, kernel_size=1, stride=1)
        self.w0 = torch.nn.Conv2d(self.lookback, self.hidden_width, kernel_size=1)
        self.convS1 = SpectralConv2d(self.hidden_width, self.hidden_width, self.modes1, self.modes2)
        self.conv1 = nn.Conv2d(self.hidden_width, self.hidden_width, kernel_size=1, stride=1)
        self.w1 = torch.nn.Conv2d(self.hidden_width, self.hidden_width, kernel_size=1)
        # self.convS2 = SpectralConv2d(self.hidden_width, self.hidden_width, self.modes1, self.modes2)
        # self.conv2 = nn.Conv2d(self.hidden_width, self.hidden_width, kernel_size=1, stride=1)
        # self.w2 = torch.nn.Conv2d(self.hidden_width, self.hidden_width, kernel_size=1)

        self.fc1 = torch.nn.Linear(self.hidden_width, self.h_dim0)
        self.fc2 = torch.nn.Linear(self.h_dim0, self.h_dim1)
        self.fc3 = torch.nn.Linear(self.h_dim1, self.h_dim1)
        self.noiseExp0_real = nn.Linear(self.h_dim1, self.output_dim)
        self.noiseExp0_imag = nn.Linear(self.h_dim1, self.output_dim)
        self.noisePhase0_real = nn.Linear(self.h_dim1, self.output_dim)
        self.noisePhase0_imag = nn.Linear(self.h_dim1, self.output_dim)
        self.noiseAmplitude0_real = nn.Linear(self.h_dim1, self.output_dim)
        self.noiseAmplitude0_imag = nn.Linear(self.h_dim1, self.output_dim)
        self.nExp0_real = torch.tensor(1., requires_grad=True)
        self.nExp0_imag = torch.tensor(1., requires_grad=True)
        self.nPhase0_real = torch.tensor(1., requires_grad=True)
        self.nPhase0_imag = torch.tensor(1., requires_grad=True)
        self.nAmp0_real = torch.tensor(1., requires_grad=True)
        self.nAmp0_imag = torch.tensor(1., requires_grad=True)

        self.noiseExp1_real = nn.Linear(self.h_dim1, self.output_dim)
        self.noiseExp1_imag = nn.Linear(self.h_dim1, self.output_dim)
        self.noisePhase1_real = nn.Linear(self.h_dim1, self.output_dim)
        self.noisePhase1_imag = nn.Linear(self.h_dim1, self.output_dim)
        self.noiseAmplitude1_real = nn.Linear(self.h_dim1, self.output_dim)
        self.noiseAmplitude1_imag = nn.Linear(self.h_dim1, self.output_dim)
        self.nExp1_real = torch.tensor(1., requires_grad=True)
        self.nExp1_imag = torch.tensor(1., requires_grad=True)
        self.nPhase1_real = torch.tensor(1., requires_grad=True)
        self.nPhase1_imag = torch.tensor(1., requires_grad=True)
        self.nAmp1_real = torch.tensor(1., requires_grad=True)
        self.nAmp1_imag = torch.tensor(1., requires_grad=True)

        # self.noiseExpA = torch.tensor(1., requires_grad=True)

        self.head1 = nn.Linear(self.h_dim1, self.output_dim)
        self.head2 = nn.Linear(self.h_dim1, self.output_dim)
        self.head3 = nn.Linear(self.h_dim1, self.output_dim)
        self.head4 = nn.Linear(self.h_dim1, self.output_dim)
        self.head5 = nn.Linear(self.h_dim1, self.output_dim)
        self.head6 = nn.Linear(self.h_dim1, self.output_dim)
        # self.head7 = nn.Linear(self.h_dim1, self.output_dim)

        self.headA = nn.Linear(self.hidden_width, self.output_dim)
        self.headB = nn.Linear(self.hidden_width, self.output_dim)
        self.headC = nn.Linear(self.hidden_width, self.output_dim)
        self.headD = nn.Linear(self.hidden_width, self.output_dim)
        self.headE = nn.Linear(self.hidden_width, self.output_dim)
        self.headF = nn.Linear(self.hidden_width, self.output_dim)
        # self.headG = nn.Linear(self.hidden_width, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_uniform(self.weight)

    def forward(self, x):
        if x.size(dim=0) == 1:
            x = torch.squeeze(x, dim=1)

        x = torch.unsqueeze(self.fc0(x), dim=3)  # Batch, Height, Width, H

        x = torch.nn.functional.pad(x, [0, self.hidden_width + self.padding, 0, 0])

        x1 = self.convS0(x, self.nExp0_real, self.nExp0_imag, self.nAmp0_real, self.nAmp0_imag, self.nPhase0_real,
                         self.nPhase0_imag)
        x1 = self.conv0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = torch.nn.functional.gelu(x)

        x1 = self.convS1(x, self.nExp1_real, self.nExp1_imag, self.nAmp1_real, self.nAmp1_imag, self.nPhase1_real,
                         self.nPhase1_imag)  # Batch, H, Height, Width
        x1 = self.conv1(x1)
        x2 = self.w1(x)  # Batch, H, Height, Width
        x = x1 + x2  # Batch, H, Height, Width
        x = torch.nn.functional.gelu(x)

        # x1 = self.convS2(x)  # Batch, H, Height, Width
        # x1 = self.conv2(x1)
        # x2 = self.w2(x)  # Batch, H, Height, Width
        # x = x1 + x2  # Batch, H, Height, Width
        # x = torch.nn.functional.gelu(x)

        if self.padding + self.hidden_width > 0:
            x = x[..., :-self.hidden_width - self.padding]
        x = x.permute(0, 2, 3, 1)  # Batch, Height, Width, H
        x = self.fc1(x)  # Batch, Height, Width, 128
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.gelu(self.fc2(x))  # Batch, Height, Width, 1
        x_noise = torch.nn.functional.gelu(self.fc3(x))

        self.nExp0_real = self.noiseExp0_real(x_noise)
        self.nExp0_imag = self.noiseExp0_imag(x_noise)
        self.nAmp0_real = self.noisePhase0_real(x_noise)
        self.nAmp0_imag = self.noisePhase0_imag(x_noise)
        self.nPhase0_real = self.noiseAmplitude0_real(x_noise)
        self.nPhase0_imag = self.noiseAmplitude0_imag(x_noise)

        self.nExp1_real = self.noiseExp1_real(x_noise)
        self.nExp1_imag = self.noiseExp1_imag(x_noise)
        self.nPhase1_real = self.noisePhase1_real(x_noise)
        self.nPhase1_imag = self.noisePhase1_imag(x_noise)
        self.nAmp1_real = self.noiseAmplitude1_real(x_noise)
        self.nAmp1_imag = self.noiseAmplitude1_imag(x_noise)

        A = torch.nn.functional.gelu(torch.squeeze(self.head1(x)))
        B = torch.nn.functional.gelu(torch.squeeze(self.head2(x)))
        C = torch.nn.functional.gelu(torch.squeeze(self.head3(x)))
        D = torch.nn.functional.gelu(torch.squeeze(self.head4(x)))
        E = torch.nn.functional.gelu(torch.squeeze(self.head5(x)))
        F = torch.nn.functional.gelu(torch.squeeze(self.head6(x)))
        # g = torch.nn.functional.gelu(torch.squeeze(self.head7(x)))

        a = self.headA(A)
        b = self.headB(B)
        c = self.headC(C)
        d = self.headD(D)
        e = self.headE(E)
        f = self.headF(F)
        # g = self.headG(g)

        # print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)
        return a, b, c, d, e, f  # , g
