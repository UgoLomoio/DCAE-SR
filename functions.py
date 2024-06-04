import torch
from torch import optim, nn
from IPython.display import clear_output
from torchinfo import summary
import neurokit2 as nk 
from sklearn.decomposition import PCA
from plotly import graph_objects as go
import os
from math import ceil 
import cv2
from matplotlib import cm
from matplotlib import rcParams
import pytorch_lightning as pl 
import imageio
import gc 
import collections
from pytorch_lightning.loggers import Logger 
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
import scipy   
import wfdb
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import math 
from torch.utils.data import DataLoader, Dataset
from scipy.signal import butter, lfilter, iirnotch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import LinearLR
from pytorch_lightning.callbacks import LearningRateMonitor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pycwt 
from torch.nn import functional as F

import matplotlib 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
font = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

#gc.collect()
#with torch.no_grad():
#    torch.cuda.empty_cache()
    
warnings.simplefilter(action='ignore', category=FutureWarning)    
rcParams['font.weight'] = 'bold'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from sys import platform

if platform == "darwin":
    if torch.cuda.is_available():
        device = "cuda"
        accelerator = "gpu"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        accelerator = "mps" if torch.backends.mps.is_available() else "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

map_superclass_rev = {'CD': 0, 'HYP': 1, 'MI': 2, 'NORM': 3, 'STTC': 4, 'All': 99}

def bandpass(lowcut, highcut, order=5, fs = 50):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_highpass(highcut, fs=50, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')#, analog = True)
    return b, a

def notch_filter(cutoff, q, fs=50):

    nyq = 0.5*fs
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a


def myfilter_hr(lowcut, highcut, data, powerline=None, fs = 500):

    nchs = 12
    filtered_data = torch.zeros_like(data)
    for ch in range(nchs):
        ch_data = data[ch, :]
        b, a = bandpass(lowcut, highcut, fs = fs)
        x = lfilter(b, a, ch_data)
        if powerline is not None:
            f, e = notch_filter(powerline, 30, fs = fs)
            x = lfilter(f, e, x)
        filtered_data[ch, :] = torch.from_numpy(x)
    return filtered_data

def myfilter(lowcut, data, powerline=None, fs = 50):

    nchs = 12
    filtered_data = torch.zeros_like(data)
    for ch in range(nchs):
        ch_data = data[ch, :]
        b, a = butter_highpass(lowcut, fs = fs)
        x = lfilter(b, a, ch_data)
        if powerline is not None:
            f, e = notch_filter(powerline, 30, fs = fs)
            x = lfilter(f, e, x)
        filtered_data[ch, :] = torch.from_numpy(x)
    return filtered_data


def plot_wavelet_reconstruction(signal, wav, scale, fig, axs, fs = 50):
    
    dt = 1/fs
    
    ch = 0
    if wav.ndim == 4:
        wav = wav[0, ch, :, :]
    elif wav.ndim == 3:
        wav = wav[ch, :, :]
    if scale.ndim == 3:
        scale = scale[0, ch, :]
    elif scale.ndim == 2:
        scale = scale[ch, :]
       
    print("Wavelets reconstruction ", wav.shape, scale.shape)
    
    signal = signal.cpu().detach().numpy()
    scale = scale.cpu().detach().numpy()
    wav = wav.cpu().detach().numpy()
    
    reconstructed = pycwt.icwt(wav, scale, dt, wavelet='morlet')
    

    if signal.ndim == 3:
        signal = signal[0, ch, :]
    elif signal.ndim == 2:
        signal = signal[ch, :]
    
    if reconstructed.ndim == 3:
        reconstructed = reconstructed[0, ch, :]
    elif reconstructed.ndim == 2:
        reconstructed = reconstructed[ch, :]
        
        
    axs.plot(signal, "g", label="original")
    axs.plot(reconstructed, "--r", label="reconstructed")
        
    fig.canvas.draw()
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig.canvas.flush_events()

def viz(Tx, Wx):

    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()

    
def plot_gradients_blocks(block, data, img, module, block_num, row, col, epoch, fig, axs, ch):
        
    pred = block(data)#(torch.unsqueeze(data, dim = 0))
    n = data.shape[-1]
    t = np.arange(0, n, 1)
    
    #print(pred.shape)

    fig.suptitle("Module {}, Channel {}".format(module, ch))
    axs[row, col].set_title("Block {}".format(block_num))
    #plot gradients 
    
    #pred.backward()
    
    #pull the gradients out of the model
    gradients = block.get_activations_gradient()
    
    if gradients is not None:
        
        print("Gradients", gradients.shape)
        
        # pool the gradients across the channels
        #pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the convolutional block
        activations = block.get_activations(data.to(device)).detach()
        channels = data.shape[0]
        del data
        
        print("Activations", activations.shape)
        # weight the channels by corresponding gradients
        #for i in range(channels): 
        #    activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=0).squeeze()
        print("Heatmap1", heatmap.shape)
        
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)
        print("Heatmap2", heatmap.shape)
            
        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.cpu().detach().numpy()
        print("Heatmap3", heatmap.shape)

        heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #print(heatmap.shape)
        heatmap = np.transpose(heatmap, (1, 0, 2))
        print(heatmap.shape, img.shape)
        
        
        # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
        black_pixels = np.where(
            (img[:, :, 0] == 0) & 
            (img[:, :, 1] == 0) & 
            (img[:, :, 2] == 0)
        )
        white_pixels = np.where(
            (img[:, :, 0] == 255) & 
            (img[:, :, 1] == 255) & 
            (img[:, :, 2] == 255)
        )
        
        cmap = plt.cm.get_cmap("jet")
        heatmap[white_pixels] = [0, 0, 0]
        superimposed_img = heatmap * 0.5 + img
        
        axs[row, col].imshow(superimposed_img, cmap = "jet")
        
        
        
        #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, fraction=.1, orientation = "horizontal")
        
        fig.canvas.draw()
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()

class PeakDetectionLoss(nn.Module):
    def __init__(self, weight_peak=1.5, weight_non_peak=1.0, interval = 10, debug = False):
        super(PeakDetectionLoss, self).__init__()
        self.weight_peak = weight_peak  # Weight for peak region loss
        self.weight_non_peak = weight_non_peak  # Weight for non-peak region loss
        self.interval = interval #interval that can contain the PQRST curve [rpeak - 60, rpeak + 60] if interval = 60
        self.debug = debug 
        
    def forward(self, x, x_recons):
        
        #print(x.shape, x_recons.shape)
        if x.ndim == 3:
            x = torch.squeeze(x, dim=0) 
        if x_recons.ndim == 3:
            x_recons = torch.squeeze(x_recons, dim=0) 
        #print(x.shape, x_recons.shape)
        
        x_rpeaks = x.cpu().detach().numpy()[0].flatten()
        rpeaks, info = nk.ecg_peaks(x_rpeaks, sampling_rate=50, method="pantompkins1985")          
        rpeaks = torch.tensor(info["ECG_R_Peaks"]).float()
        #print(rpeaks)
        
        # Loss for peak regions (encourage peak detection)
        peaks_losses = []
        non_peaks_losses = []
        Ns = {"peaks": [], "non_peaks": []}
        
        for i, peak in enumerate(rpeaks):
            
            peak = peak.item()
            if i == 0:
                t0 = int(peak-self.interval)
                if t0 > 0:
                    #print("init", t0)
                    #print(x.shape, x_recons.shape)
                    first_loss = torch.pow(x[:, :t0] - x_recons[:, :t0], 2).flatten()
                    Ns["non_peaks"].append(t0) 
                    #print(first_loss)
                    for elem in first_loss:
                        non_peaks_losses.append(elem)
                        
                t1 = int(peak+self.interval)
                if i+1 < len(rpeaks):
                    t2 = int(rpeaks[i+1]-self.interval)
                    if t2 > t1:
                        #print("between", t1, t2)
                        between_peak_loss = torch.pow(x[:, t1:t2] - x_recons[:, t1:t2], 2).flatten()
                    elif t2 < t1:
                        #print("between", t2, t1)
                        between_peak_loss = torch.pow(x[:, t2:t1] - x_recons[:, t2:t1], 2).flatten()
                    #print(between_peak_loss)
                    Ns["non_peaks"].append(abs(t1-t2))
                    for elem in between_peak_loss:
                        non_peaks_losses.append(elem)
                    
            elif i < len(rpeaks)-1:
                t1 = int(peak+self.interval)
                t2 = int(rpeaks[i+1]-self.interval)
                if t2 > t1:
                    #print("between", t1, t2)
                    between_peak_loss = torch.pow(x[:, t1:t2] - x_recons[:, t1:t2], 2).flatten()
                elif t2 < t1: #if t2 == t1, dont do nothing, skip this part
                    #print("between", t2, t1)
                    between_peak_loss = torch.pow(x[:, t2:t1] - x_recons[:, t2:t1], 2).flatten()
                #print(between_peak_loss)
                Ns["non_peaks"].append(abs(t1-t2))
                for elem in between_peak_loss:
                    non_peaks_losses.append(elem)
               
            else:
                t3 = int(peak+self.interval)
                if t3 < 300:
                    #print("last",t3)
                    last_loss = torch.pow(x[:, t3:] - x_recons[:, t3:], 2).flatten()
                    #print(last_loss)
                    Ns["non_peaks"].append(300-t3) 
                    for elem in last_loss:
                        non_peaks_losses.append(elem)
             
            t4 = int(peak-self.interval)
            t5 = int(peak+self.interval)
            if t5 >= 300:
                t5 = 300
            #print("PEAK", t4, t5)
            Ns["peaks"].append(self.interval*2)#abs(t5-t4)
            loss = torch.pow(x[:, t4:t5] - x_recons[:, t4:t5], 2).flatten()
            #print(loss)
            for elem in loss:
                peaks_losses.append(elem)
        
        #print(peaks_losses, non_peaks_losses)

        peak_loss = 0.5*torch.mean(torch.tensor(peaks_losses))#torch.sum(torch.tensor(peaks_losses))/np.sum(Ns["peaks"])
        non_peak_loss = 0.5*torch.mean(torch.tensor(non_peaks_losses))#torch.sum(torch.tensor(non_peaks_losses))/np.sum(Ns["non_peaks"])
        total_loss = peak_loss*self.weight_peak + non_peak_loss*self.weight_non_peak
        #print(peak_loss, non_peak_loss, total_loss)
        if not self.debug:
            total_loss.requires_grad_()
            return total_loss
        else:

            return peaks_losses, non_peaks_losses, Ns
            #return peak_loss, non_peak_loss, total_loss



class EncoderBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do):
        super(EncoderBlock1d, self).__init__()
        
        
        #self.conv1d_in = nn.Conv1d(in_channels, out_channels, kernel_size=ks, stride=stride, bias=True)
        #self.activation_in = activation_layer(activation_name)
        
        self.conv1d_in = nn.Conv1d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], bias=True)
        self.activation_in = activation_layer(activation_name)
        #self.batch_norm_in = nn.BatchNorm1d(out_channels[0])
        self.dropout_in = nn.Dropout(p=do)
        self.conv1d_out = nn.Conv1d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], bias=True)
        #self.batch_norm_out = nn.BatchNorm1d(out_channels[1])
        self.activation_out = activation_layer(activation_name)
        self.dropout_out = nn.Dropout(p=do)
   
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv1d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv1d_out(x)
        #x = self.batch_norm_out(x)
        x = self.activation_out(x)
        x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x


class DecoderBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do, last = False, last_tanh = True):
        super(DecoderBlock1d, self).__init__()
        
        #self.conv1d_in = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=ks, stride=stride, bias=True)
        #self.activation_in = activation_layer(activation_name)
        self.last = last
        self.last_tanh = last_tanh

        self.conv1d_in = nn.ConvTranspose1d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], bias=True)
        self.activation_in = activation_layer(activation_name)
        #self.batch_norm_in = nn.BatchNorm1d(out_channels[0])
        self.dropout_in = nn.Dropout(p=do)
        self.conv1d_out = nn.ConvTranspose1d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], bias=True)
        #self.batch_norm_out = nn.BatchNorm1d(out_channels[1])
        if self.last_tanh:
            self.activation_out = activation_layer(activation_name)
        if not self.last:
            self.dropout_out = nn.Dropout(p=do)
            
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv1d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv1d_out(x)
        #x = self.batch_norm_out(x)
        
        #x = self.activation_out(x)  try this inside the if last
        if self.last_tanh:
            x = self.activation_out(x)
        if not self.last:
            x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x
    
class DecoderBlockUpsample1d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do):
        super(DecoderBlockUpsample1d, self).__init__()
        
        #self.conv1d_in = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=ks, stride=stride, bias=True)
        #self.activation_in = activation_layer(activation_name)
        
        self.conv1d_in = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=ks, stride=stride, bias=True)
        self.dropout_in = nn.Dropout(p=do)
        self.activation_in = activation_layer(activation_name)
        #dimshuffle
        #stacking
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv1d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv1d_out(x)
        #x = self.batch_norm_out(x)
        x = self.activation_out(x)
        x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x

class DeconvUpsampler2d(nn.Module):
    def __init__(self, nchs, height, width, scale_factor, activation_name):
        super(DeconvUpsampler2d, self).__init__()
        
        self.nchs = nchs
        self.width = width
        self.height = height
        self.scale_factor = int(scale_factor)
        self.activation_name = activation_name
        
        self.wanted_width = self.width * self.scale_factor
        self.wanted_height = self.height * self.scale_factor
        self.size = (self.wanted_width, self.wanted_height)
        
        self.n_layers = self.scale_factor
        
        self.kernel_sizes = [100 for i in range(self.n_layers)]
        self.strides = [1 for i in range(self.n_layers)]
        
        self.blocks = []

        for i in range(self.n_layers):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            print("AAAA up", ks, stride)
            self.blocks.append(DecoderBlockUpsample2d(nchs, nchs, ks, stride, self.activation_name))
        
        self.upsampler = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        #print("Upsample")
        #print(x.shape)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        #self.plot_activations()
           
        return x
    
class DeconvUpsampler1d(nn.Module):
    def __init__(self, nchs, height, width, scale_factor, activation_name):
        super(DeconvUpsampler1d, self).__init__()
        
        self.nchs = nchs
        self.width = width
        self.height = height
        self.scale_factor = int(scale_factor)
        self.activation_name = activation_name
        
        self.wanted_width = self.width * self.scale_factor
        self.size = self.wanted_width
        
        self.n_layers = self.scale_factor
        
        self.kernel_sizes = [100 for i in range(self.n_layers)]
        self.strides = [1 for i in range(self.n_layers)]
        
        self.blocks = []

        for i in range(self.n_layers):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            print("AAAA up", ks, stride)
            self.blocks.append(DecoderBlockUpsample1d(nchs, nchs, ks, stride, self.activation_name))
        
        self.upsampler = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        #print("Upsample")
        #print(x.shape)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        #self.plot_activations()
           
        return x

class EncoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do):
        super(EncoderBlock2d, self).__init__()
        
        self.conv2d_in = nn.Conv2d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], padding = 1, bias=True)
        self.activation_in = activation_layer(activation_name)
        #self.batch_norm_in = nn.BatchNorm2d(out_channels[0])
        self.dropout_in = nn.Dropout(p=do)
        self.conv2d_out = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
        #self.batch_norm_out = nn.BatchNorm2d(out_channels[1])
        self.activation_out = activation_layer(activation_name)
        self.dropout_out = nn.Dropout(p=do)
   
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv2d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv2d_out(x)
        #x = self.batch_norm_out(x)
        x = self.activation_out(x)
        x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x


class DecoderBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do, last = False):
        super(DecoderBlock2d, self).__init__()
        
        self.last = last
        
        if self.last: 
            self.conv2d_in = nn.ConvTranspose2d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], padding = 1, bias=True)
        else:
            self.conv2d_in = nn.ConvTranspose2d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], padding = 1, bias=True)
        self.activation_in = activation_layer(activation_name)
        #self.batch_norm_in = nn.BatchNorm2d(out_channels[0])
        self.dropout_in = nn.Dropout(p=do)
        if self.last:  
            self.conv2d_out = nn.ConvTranspose2d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
        else:
            self.conv2d_out = nn.ConvTranspose2d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
        #self.batch_norm_out = nn.BatchNorm2d(out_channels[1])
        self.activation_out = activation_layer(activation_name)
        if not self.last:
            self.dropout_out = nn.Dropout(p=do)
            
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv2d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv2d_out(x)
        #x = self.batch_norm_out(x)
        x = self.activation_out(x)
        if not self.last:
            x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x
    

class DecoderBlock2dSR(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do, last = False):
        super(DecoderBlock2dSR, self).__init__()
        
        self.last = last

        self.conv2d_in = nn.ConvTranspose2d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], padding = 1, bias=True)        
        self.dropout_in = nn.Dropout(p=do)
        self.activation_in = activation_layer(activation_name)
        
        self.conv2d_out = nn.ConvTranspose2d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
        if not self.last:
            self.dropout_out = nn.Dropout(p=do)
        self.activation_out = activation_layer(activation_name)
        #dimshuffle
        #stacking
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv2d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)

        x = self.conv2d_out(x)
        x = self.activation_out(x)
        #x = self.batch_norm_out(x)
        if not self.last:
            x = self.dropout_out(x)

        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x
    

class DecoderBlock1dSR(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do, last = False, last_tanh = True):
        super(DecoderBlock1dSR, self).__init__()
        self.last_tanh = last_tanh
        self.last = last
        if stride[0] != 1:#deleted out_padding
            self.conv1d_in = nn.ConvTranspose1d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], padding = 1, bias=True)
        else:
            self.conv1d_in = nn.ConvTranspose1d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], padding = 1, bias=True)
        
        self.dropout_in = nn.Dropout(p=do)
        self.activation_in = activation_layer(activation_name)
        
        if self.last: 
            if stride[1] != 1:
                self.conv1d_out = nn.ConvTranspose1d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
            else:
                self.conv1d_out = nn.ConvTranspose1d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
            
        else:
            self.conv1d_out = nn.ConvTranspose1d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], padding = 1, bias=True)
        if not self.last:
            self.dropout_out = nn.Dropout(p=do)
        if self.last_tanh:    
            self.activation_out = activation_layer(activation_name)

        #dimshuffle
        #stacking
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv1d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)

        x = self.conv1d_out(x)
        #x = self.activation_out(x)#try this inside the if last
        #x = self.batch_norm_out(x)

        if self.last_tanh:   
            x = self.activation_out(x)
        if not self.last:
            x = self.dropout_out(x)

        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x


class EncoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do):
        super(EncoderBlock3d, self).__init__()
        
        self.conv3d_in = nn.Conv3d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], bias=True)
        self.activation_in = activation_layer(activation_name)
        #self.batch_norm_in = nn.BatchNorm3d(out_channels[0])
        self.dropout_in = nn.Dropout(p=do)
        self.conv3d_out = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], bias=True)
        #self.batch_norm_out = nn.BatchNorm3d(out_channels[1])
        self.activation_out = activation_layer(activation_name)
        self.dropout_out = nn.Dropout(p=do)
   
        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv3d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv3d_out(x)
        #x = self.batch_norm_out(x)
        x = self.activation_out(x)
        x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do, last = False):
        super(DecoderBlock3d, self).__init__()
        
        self.last = last 
        
        self.conv3d_in = nn.ConvTranspose3d(in_channels, out_channels[0], kernel_size=ks[0], stride=stride[0], bias=True)
        self.activation_in = activation_layer(activation_name)
        #self.batch_norm_in = nn.BatchNorm3d(out_channels[0])
        self.dropout_in = nn.Dropout(p=do)
        self.conv3d_out = nn.ConvTranspose3d(out_channels[0], out_channels[1], kernel_size=ks[1], stride=stride[1], bias=True)
        #self.batch_norm_out = nn.BatchNorm3d(out_channels[1])
        self.activation_out = activation_layer(activation_name)
        if not self.last:
            self.dropout_out = nn.Dropout(p=do)
            
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv3d_in(x)
        x = self.activation_in(x)
        #x = self.batch_norm_in(x)
        x = self.dropout_in(x)
        x = self.conv3d_out(x)
        #x = self.batch_norm_out(x)
        x = self.activation_out(x)
        if not self.last:
            x = self.dropout_out(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x
    
class DecoderBlockUpsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, activation_name, do):
        super(DecoderBlockUpsample3d, self).__init__()
        
        
        self.conv3d_in = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=ks, stride=stride, bias=True)
        self.dropout_in = nn.Dropout(p=do)
        self.activation_in = activation_layer(activation_name)
        #dimshuffle
        #stacking
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv3d_in(x)
        x = self.dropout_in(x)
        x = self.activation_in(x)
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
        return x


class Encoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img):
        super(Encoder1d, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        
        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
        print(nrows, ncols)
        
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio

        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)

        
        self.blocks = []
        
        for i, out_channel in enumerate(out_channels):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            print("AAAA", ks, stride)
            self.blocks.append(EncoderBlock1d(in_channels, out_channel, ks, stride, self.activation_name, self.do))
            in_channels = out_channel[-1]
        
        # placeholder for the gradients
        self.gradients = None
        
        self.encoder = nn.Sequential(*self.blocks)
        #self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
                temp = self.to_predict.detach().clone()
           
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            print(row, col)
            plot_gradients_blocks(block, prev, self.img, "Encoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Encode")
        #print(x.shape)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        #self.plot_activations()
           
        return x

class Decoder1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img, last_tanh = True):
        super(Decoder1d, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        self.last_tanh = last_tanh

        self.num_layers = len(self.out_channels)
        self.nrows = 2
        self.ncols = ceil(self.num_layers/ self.nrows)
 
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
        self.ax = [self.fig.add_subplot( self.nrows, self.ncols, i+1) for i in range(self.nrows*self.ncols)]
        self.ax = np.array(self.ax).reshape( self.nrows, self.ncols)
        
        
        self.blocks = []
        
        for i, out_channel in enumerate(out_channels):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            if i < len(out_channels)-1:
                self.blocks.append(DecoderBlock1d(in_channels, out_channel, ks, stride, self.activation_name, self.do, last_tanh = self.last_tanh))
            else:
                self.blocks.append(DecoderBlock1d(in_channels, out_channel, ks, stride, "tanh", self.do, last = True, last_tanh = self.last_tanh))
            in_channels = out_channel[-1]
            
        self.gradients = None
        self.decoder = nn.Sequential(*self.blocks)
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i -  self.ncols
            
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            plot_gradients_blocks(block, prev, self.img, "Decoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Decode")
        #print(x.shape)

        for i, block in enumerate(self.blocks):
            x = block(x)
      
            # register the hook
            #if x.requires_grad:
            #    h = x.register_hook(self.activations_hook)
                        
        #self.plot_activations()
            
        return x

class Encoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img):
        super(Encoder2d, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        
        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
        print(nrows, ncols)
        
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio

        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)

        
        self.blocks = []
        
        for i, out_channel in enumerate(out_channels):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            print("AAAA", ks, stride)
            self.blocks.append(EncoderBlock2d(in_channels, out_channel, ks, stride, self.activation_name, self.do))
            in_channels = out_channel[-1]
        
        # placeholder for the gradients
        self.gradients = None
        
        self.encoder = nn.Sequential(*self.blocks)
        #self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
                temp = self.to_predict.detach().clone()
           
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            print(row, col)
            plot_gradients_blocks(block, prev, self.img, "Encoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Encode")
        #print(x.shape)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        #self.plot_activations()
           
        return x


class Decoder2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img):
        super(Decoder2d, self).__init__()
        
        print(in_channels, out_channels, kernel_sizes, strides)
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        
        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
 
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)
        
        
        self.blocks = []
        
        for i, out_channel in enumerate(out_channels):

            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            if i < len(out_channels)-1:
                self.blocks.append(DecoderBlock2d(in_channels, out_channel, ks, stride, self.activation_name, self.do))
            else:
                self.blocks.append(DecoderBlock2d(in_channels, out_channel, ks, stride, self.activation_name, self.do, last = True))
            in_channels = out_channel[-1]
            
        self.gradients = None
        self.decoder = nn.Sequential(*self.blocks)
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
            
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            plot_gradients_blocks(block, prev, self.img, "Decoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Decode")
        #print(x.shape)

        for i, block in enumerate(self.blocks):
            x = block(x)
      
            # register the hook
            #if x.requires_grad:
            #    h = x.register_hook(self.activations_hook)
                        
        #self.plot_activations()            
        return x


class Decoder2dSR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img):
        super(Decoder2dSR, self).__init__()
        
        print(in_channels, out_channels, kernel_sizes, strides)
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        
        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
 
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)
        
        
        self.blocks = []  
        
        for i, out_channel in enumerate(out_channels):

            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            if (i < len(out_channels)-1):
                self.blocks.append(DecoderBlock2dSR(in_channels, out_channel, ks, stride, self.activation_name, self.do))
            else:
                self.blocks.append(DecoderBlock2dSR(in_channels, out_channel, ks, stride, self.activation_name, self.do, last = True))
            in_channels = out_channel[-1]
            
        self.gradients = None
        self.decoder = nn.Sequential(*self.blocks)
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
            
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            plot_gradients_blocks(block, prev, self.img, "Decoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Decode")
        #print(x.shape)

        for i, block in enumerate(self.blocks):
            x = block(x)
      
            # register the hook
            #if x.requires_grad:
            #    h = x.register_hook(self.activations_hook)
                        
        #self.plot_activations()            
        return x



class Decoder1dSR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img, last_tanh = True):
        super(Decoder1dSR, self).__init__()
        
        print(in_channels, out_channels, kernel_sizes, strides)
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        self.last_tanh = last_tanh

        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
 
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)
        
        
        self.blocks = []  
        
        for i, out_channel in enumerate(out_channels):

            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            if (i < len(out_channels)-1):
                self.blocks.append(DecoderBlock1dSR(in_channels, out_channel, ks, stride, self.activation_name, self.do, last_tanh = self.last_tanh))
            else:
                self.blocks.append(DecoderBlock1dSR(in_channels, out_channel, ks, stride, self.activation_name, self.do, last = True, last_tanh = self.last_tanh))
            in_channels = out_channel[-1]
            
        self.gradients = None
        self.decoder = nn.Sequential(*self.blocks)
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
            
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            plot_gradients_blocks(block, prev, self.img, "Decoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Decode")
        #print(x.shape)

        for i, block in enumerate(self.blocks):
            x = block(x)
      
            # register the hook
            #if x.requires_grad:
            #    h = x.register_hook(self.activations_hook)
                        
        #self.plot_activations()            
        return x


class Encoder3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img):
        super(Encoder3d, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        
        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
        print(nrows, ncols)
        
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio

        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)

        
        self.blocks = []
        
        for i, out_channel in enumerate(out_channels):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            print("AAAA", ks, stride)
            self.blocks.append(EncoderBlock3d(in_channels, out_channel, ks, stride, self.activation_name, self.do))
            in_channels = out_channel[-1]
        
        # placeholder for the gradients
        self.gradients = None
        
        self.encoder = nn.Sequential(*self.blocks)
        #self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
                temp = self.to_predict.detach().clone()
           
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            print(row, col)
            plot_gradients_blocks(block, prev, self.img, "Encoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Encode")
        #print(x.shape)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            
        #self.plot_activations()
           
        return x

class Decoder3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, do, activation_name, to_predict, img):
        super(Decoder3d, self).__init__()
        
        print(in_channels, out_channels, kernel_sizes, strides)
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.do = do
        self.activation_name = activation_name
        self.to_predict = to_predict
        self.img = img 
        
        self.num_layers = len(self.out_channels)
        nrows = 2
        ncols = ceil(self.num_layers/nrows)
 
        self.fig = plt.figure(figsize=(8,8)) # Notice the equal aspect ratio
        self.ax = [self.fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        self.ax = np.array(self.ax).reshape(nrows, ncols)
        
        
        self.blocks = []
        
        for i, out_channel in enumerate(out_channels):

            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            if i < len(out_channels)-1:
                self.blocks.append(DecoderBlock3d(in_channels, out_channel, ks, stride, self.activation_name, self.do))
            else:
                self.blocks.append(DecoderBlock3d(in_channels, out_channel, ks, stride, self.activation_name, self.do, last = True))
            in_channels = out_channel[-1]
            
        self.gradients = None
        self.decoder = nn.Sequential(*self.blocks)
        
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x, i):
        return self.blocks[i](x)
    
    def plot_activations(self):
         
        temp = self.to_predict.detach().clone()

        for i, block in enumerate(self.blocks):
            if i < self.ncols:
                row = 0
                col = i    
            else:
                row = 1
                col = i - self.ncols
            
            prev = temp.detach().clone()
            temp = block(temp)
            print(block, prev.shape, temp.shape)
            # register the hook
            if temp.requires_grad:
                h = temp.register_hook(self.activations_hook)
            plot_gradients_blocks(block, prev, self.img, "Decoder", i, row, col, "None", self.fig, self.ax)
    
    def forward(self, x):
        #print("Decode")
        #print(x.shape)

        for i, block in enumerate(self.blocks):
            x = block(x)
      
            # register the hook
            #if x.requires_grad:
            #    h = x.register_hook(self.activations_hook)
                        
        #self.plot_activations()
            
        return x

def reverse_listoflist(list):
    rev = [x[::-1] for x in list[::-1]]
    return rev

class VAE1d_SR_multimodal(nn.Module):

    """
    1 Dimensional VAE with SUPER RESOLUTION.
    The network is made up by stacking ENASMacroLayer. The Macro search space contains these layers.
    Each layer chooses an operation from predefined ones and SkipConnect then forms a network.

    Parameters
    ---
    num_layers: int
        The number of layers contained in the network.
    in_channel: int
        The number of input's channels.
    num_classes: int
        The number of classes for classification.
    dropout_rate: float
        Dropout layer's dropout rate before the final dense layer.
    """
    def __init__(self, to_predict, img, num_layers=3, out_filters=None, in_channels=12, kernel_sizes = None, strides = None, do = 0.2, batch_size = 32, loss = "mse",
                 scale_factor = None, size = 500, latent_dim = 4096, activation_functions = None, width = 250, device = torch.device("cpu"), 
                 mode = "s", activation_name = "tanh", loss_name="mse", type = "ae", sr_type = "convt", str_sr = None, of_sr = None, kernel_sizes_sr = None, last_tanh = True, loss_type = "lr+hr", denoising = True):
        """
        mode: 
            s - reconstruction error only signal
            w - reconstruction error only wavelets
            sw/ws - reconstruction error signal and wavelets
        """         
        
        super().__init__()

        self.supported_modes = ["s"]
        self.supported_types = ["ae", "vae"]
        self.supported_sr_types = ["upsample", "convt", "none", None]
        self.supported_loss_types = ["lr", "lr+hr", "hr"]
        
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.out_filters = out_filters
        self.in_channels = in_channels
        self.do = do 
        self.scale_factor = scale_factor
        self.kernel_sizes = kernel_sizes
        self.strides = strides 
        self.width = width 
        self.to_predict = to_predict 
        self.img = img 
        self.loss_name = loss_name
        self.denoising = denoising
        self.last_tanh = last_tanh

        if sr_type == "none":
            sr_type = None
        elif sr_type not in self.supported_sr_types:
            sr_type = None
        self.sr_type = sr_type

        if loss_type not in self.supported_loss_types:
            if self.sr_type is None:
                loss_type = "lr"
            else:
                loss_type = "lr+hr"
        self.loss_type = loss_type 
        
        
        if mode not in self.supported_modes:
            raise Exception("Mode {} for Single Modality Super Resolution 1d DCAE not supported. Supported modalities: 's' for only signal reconstruction error".format(mode))
        
        self.mode = mode 
            
        
        if size is not None:
            self.size = size 
        else:
            self.size = width * 10
            
        self.device = device
        self.decoder_input_shape = None
        self.x_input_shape = self.in_channels

        if self.out_filters is None:
            self.out_filters = [int(in_channels/(i+1)) for i in range(self.num_layers)]
            self.out_filters[-1] = 1
        if self.kernel_sizes is None:
            self.kernel_sizes = [1 for i in range(self.num_layers)]
        if self.strides is None:
            self.strides = [1 for i in range(self.num_layers)]
        
        toprod =  [item for items in self.kernel_sizes for item in items]#item*2
        div = np.sum([elem for elem in toprod if elem >= 1]) 

        self.y_input_shape = int(self.width - div) 
        self.latent_dim_nodense = (1 * self.y_input_shape + self.num_layers*2)#self.num_layers * 2
        #self.latent_dim = latent_dim 
        
        self.type = type.lower()
        if self.type not in self.supported_types:
            self.type = "ae"
        
        self.loss = loss
        self.activation_name = activation_name
        
        #print(self.latent_dim_nodense, self.latent_dim)
        """
        modules= []
        in_channel = self.in_channels
        for i, out_channel in enumerate(self.out_filters):
            ks = self.kernel_sizes[i]
            stride = self.strides[i]
            modules.append(EncoderBlock1d(in_channel, out_channel, ks, stride, activation_name))

            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channels=out_channel,
                              kernel_size= ks, stride= stride, padding=0, bias=True),
                    #nn.BatchNorm1d(out_channel),
                    activation_layer(activation_name),
                    nn.Conv1d(out_channel, out_channels=out_channel,
                              kernel_size= ks, stride= stride, padding=0, bias=True),
                    #nn.BatchNorm1d(out_channel),
                    activation_layer(activation_name),
                ).to(self.device)
            )
            #in_channel = out_channel
        """
        input_linear = self.latent_dim_nodense
        
        with torch.no_grad():
            
            torch.cuda.empty_cache()
            input_shape = (self.batch_size, self.in_channels, self.width) 
            
            if "s" in self.mode:
                #self.encoder = nn.Sequential(*modules).to(self.device)
                print("Encoder ks: ", self.kernel_sizes)
                self.encoder = Encoder1d(self.in_channels, self.out_filters, self.kernel_sizes, self.strides, self.do, self.activation_name, self.to_predict, self.img).to(self.device)
                #print(self.encoder)
                print("Encoder ECG: ", summary(self.encoder, input_shape), "\n")
     
            self.encoder_out = self.latent_dim_nodense*self.batch_size* self.out_filters[-1][-1]
            #self.latent_dim = self.latent_dim_nodense*self.out_filters[-1][-1]
            
            if self.type == "vae":
                #self.latent_dim = self.latent_dim_nodense*self.out_filters[-1][-1]
                self.latent_dim = latent_dim
                print("Mu/Var", self.latent_dim_nodense*self.out_filters[-1][-1], self.latent_dim)
                self.fc_mu = nn.Linear(self.latent_dim_nodense*self.out_filters[-1][-1], self.latent_dim).to(self.device)
                self.fc_var = nn.Linear(self.latent_dim_nodense*self.out_filters[-1][-1], self.latent_dim)
            else:
                self.latent_dim = self.encoder_out
            torch.cuda.empty_cache()
        
        """    
        modules= []
        modules_sr = []
        in_channel = self.out_filters[-1]
        for i, out_channel in enumerate(self.out_filters[::-1]):
            ks = self.kernel_sizes[::-1][i]
            stride = self.strides[::-1][i]
            modules.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(in_channel, out_channels=out_channel,
                                  kernel_size= ks, stride= stride, padding=0, bias=True),
                        #nn.BatchNorm1d(out_channel),
                        activation_layer(activation_name),
                        nn.ConvTranspose1d(out_channel, out_channels=out_channel,
                                  kernel_size= ks, stride= stride, padding=0, bias=True),
                        #nn.BatchNorm1d(out_channel),
                        activation_layer(activation_name),
                        
                    ).to(self.device)
            )
            modules_sr.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(in_channel, out_channels=out_channel,
                                  kernel_size= ks, stride= stride, padding=0, bias=True),
                        #nn.BatchNorm1d(out_channel),
                        activation_layer(activation_name),
                        nn.ConvTranspose1d(out_channel, out_channels=out_channel,
                                  kernel_size= ks, stride= stride, padding=0, bias=True),
                        #nn.BatchNorm1d(out_channel),
                    ).to(self.device)
            )
        """
        
        with torch.no_grad():
            
            torch.cuda.empty_cache()
            print("Width decoder: {}/{} = {}".format(self.latent_dim, self.out_filters[-1][-1],  self.latent_dim_nodense))
            self.decoder_input_shape = (self.batch_size, self.out_filters[-1][-1], self.latent_dim_nodense)
            print(self.decoder_input_shape)

            if "s" in self.mode:
                #self.decoder = nn.Sequential(*modules).to(self.device)
                self.ks_dec = reverse_listoflist(self.kernel_sizes)
                self.of_dec = reverse_listoflist(self.out_filters)
                self.str_dec = reverse_listoflist(self.strides)
                print("Decoder ks: ", self.ks_dec)
                self.decoder = Decoder1d(self.out_filters[-1][-1], self.of_dec, self.ks_dec, self.str_dec, self.do, self.activation_name, self.to_predict, self.img, last_tanh = self.last_tanh).to(self.device)
                print("Decoder ECG no SuperResolution: ", summary(self.decoder, self.decoder_input_shape), "\n")
                    
                if self.sr_type == "upsample":
                    if self.scale_factor is not None:
                        self.upsample = nn.Upsample(scale_factor = self.scale_factor, mode="linear")
                    else:
                        self.upsample = nn.Upsample(size = self.size, mode="linear")
                
                elif self.sr_type == "convt":    
                    if of_sr is not None:
                        self.of_sr = of_sr
                    else:
                        self.of_sr = self.of_dec
                    if kernel_sizes_sr is not None:
                        self.ks_sr = kernel_sizes_sr
                    else:
                        self.ks_sr = self.ks_dec
                    if str_sr is not None:
                        self.str_sr = str_sr
                    else:
                        self.str_sr = self.str_dec 
                    self.upsample = Decoder1dSR(self.out_filters[-1][-1], self.of_sr, self.ks_sr, self.str_sr, self.do, self.activation_name, self.to_predict, self.img, last_tanh = self.last_tanh).to(self.device)
                    self.decoder_sr_input_shape =  self.decoder_input_shape
                    print("Super Resolution ECG decoder:", summary(self.upsample, self.decoder_sr_input_shape), "\n")
                
                
                if self.sr_type == "upsample":
                    summary_decoder_sr = ''' \n
                    ---------------------------------------------------- 

                    Layer (type)               Output Shape            
                    ====================================================
                
                    '''  
                    for layer in range(self.num_layers):
                        if layer != 0:
                            i = layer+1
                        else:
                            i = layer+1
                            prec = self.width - np.sum(self.kernel_sizes) #+ self.num_layers
                        
                        
                        w = int((prec+np.sum(self.kernel_sizes[layer]))) 
                        prec = w
                        ch = self.of_dec[layer] 
                        summary_decoder_sr += "   Conv1DTranspose-{}                    [-1, {}, {}] \n".format(i, ch, w) 
                        if layer != self.num_layers-1:
                            summary_decoder_sr += "    {}-{}                     [-1, {}, {}] \n".format(activation_name.upper(), i, ch, w)
                        
                    if self.scale_factor is not None:
                        w = int(prec*self.scale_factor)
                    else:
                        w = self.size 
                    summary_decoder_sr += "   Upsample-{}                  [-1, {}, {}] \n".format(i, ch, w)
                    print("Decoder ECG with SuperResolution: \n", summary_decoder_sr, "\n")
                
            torch.cuda.empty_cache()
            
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x H x W x C]
        :return: (Tensor) List of latent codes
        """
        torch.cuda.empty_cache()

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        result_ecg = None
      
        if "s" in self.mode:
            if x.ndim == 4:
                x = torch.squeeze(x, dim=1)
            #print(x.shape)
            result_ecg = self.encoder(x)
            result_ecg = torch.flatten(result_ecg)
            
            #result_ecg = result_ecg.view(-1, self.latent_dim_nodense) 

        if self.type == "vae":
            mu = self.fc_mu(result_ecg)
            log_var = self.fc_var(result_ecg)
            return [mu, log_var]
        else:
            z = result_ecg
            return z


    def decode(self, z):
        """
        NO SUPER RESOLUTION DECODER 
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x H x W x C]
        """
        torch.cuda.empty_cache()
        
        result_ecg = None
 
        if "s" in self.mode:
            result_ecg = self.decoder(z)

        return result_ecg
    
    
    def decode_sr(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x H x W x C]
        """
        torch.cuda.empty_cache()
        
        result_ecg_sr = None
        if "s" in self.mode:
            if self.sr_type == "convt":
                result_ecg_sr = self.upsample(z)
            else:
                result_ecg = self.decoder(z)
                #print(result_ecg.shape)
                #result_ecg = torch.unsqueeze(result_ecg, dim=0)
                #print(result_ecg.shape)
                result_ecg_sr = self.upsample(result_ecg)
        return result_ecg_sr
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def kl_divergence(self, z, mu, log_var):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        #std = torch.exp(0.5 * log_var)
        #p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        #q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        #log_qzx = q.log_prob(z)
        #log_pz = p.log_prob(z)

        # kl
        #kl = (log_qzx - log_pz)
        #kl = kl.sum(-1)
        
        kl = -0.5 * torch.sum(1 + log_var - torch.pow(mu, 2) - torch.exp(log_var))
        return kl
    
    def forward(self, x):
        
        #print("Input", x.shape)
        #print(len(x))
        if "hr" in self.loss_type:
            x_s, x_s_hr = x[:2]
            x_return = [x_s, x_s_hr]
        else:
            if not self.denoising:
                x_s = x[0]
                x_return = x_s
            else:#denoising low resolution 
                x_s, x_s_clean = x[:2]
                x_return = [x_s, x_s_clean]
                
        if self.type == "vae":
            mu, log_var = self.encode(x_s) #for VAE
            z = self.reparameterize(mu, log_var)
        else:
            z = self.encode(x_s)
        #print("Encoded ", z.shape)
        
        z = torch.unsqueeze(z, dim=0)
        #print("z after unsqueeze 0 dim", z.shape)
        #print("trying reshape", self.batch_size, self.out_filters[-1][-1],  self.latent_dim_nodense)
        if self.type == "vae":
            z = torch.reshape(z, (self.batch_size, self.out_filters[-1][-1],  int(self.latent_dim/self.out_filters[-1][-1]))) 
        else:
            z = torch.reshape(z, (self.batch_size, self.out_filters[-1][-1],  self.latent_dim_nodense))        
        #print("Input Decoder ", z.shape)
        decoded = self.decode(z)
        decoded = torch.squeeze(decoded, dim = 0)
        #print("Decoded ", decoded.shape)
        if self.sr_type is not None:
            decoded_sr = self.decode_sr(z)
            decoded_sr = torch.squeeze(decoded_sr, dim=0)
        else:
            decoded_sr = None
        #print("Decoded SuperResolution ", decoded_sr.shape)

        if self.type == "vae":
            return  [decoded, decoded_sr, x_return, mu, log_var] #vae
        else:
            return [decoded, decoded_sr, x_return, z]#ae

    def predict(self, x):
        
        self.eval()
        self.forward(x)
    
    def loss_function(self,
                      *args,
                    **kwargs):
        """
        Computes the Multimodal VAE with super resolution loss function.
        mse
        :param args:
        :param kwargs:
        :return:
        """
        torch.cuda.empty_cache()
        x = args[2]
        
        #print("a ", args)
        #print("k ", kwargs)
        kld_weight = kwargs["kld_weight"]
        
        if "s" in self.mode:
            
            if "hr" in self.loss_type:
                x_s, x_s_hr = x                    
            else:
                if self.denoising:
                    x_s, x_s_clean = x
                else:
                    x_s = x
            recons_s = args[0]
            recons_s = torch.unsqueeze(recons_s, dim=0)

        super_res_s = args[1]
        
        if self.type == "vae":
            mu = args[3]
            log_var = args[4]

            #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
            #print(kld_weight, "KLD weight")
            #kld_loss = - 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())# torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 0), dim = 0)#torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
             
            #expectation under z of the kl divergence between q(z|x) and
            #a standard normal distribution of the same shape
            z = self.reparameterize(mu, log_var)
            kl = self.kl_divergence(z, mu, log_var)
            
        loss_f = get_loss_function(self.loss_name) 
        if "s" in self.mode:
            
            if self.denoising:
                if self.sr_type is not None:
                    if "hr" in self.loss_type:
                        recons_loss_s = loss_f(recons_s, x_s)
                else:
                    recons_loss_s = loss_f(recons_s, x_s_clean)
            else:
                recons_loss_s = loss_f(recons_s, x_s)
                
            if self.type == "vae":
                #loss_s = torch.mean(recons_loss_s + kld_weight * kld_loss)#.item()
                #elbo
                loss_s = recons_loss_s + kld_weight * kl
            else:
                loss_s = recons_loss_s
                   
            if self.sr_type == "convt":
                if "hr" in self.loss_type:
                    
                    super_res_loss_s = loss_f(super_res_s, x_s_hr)
                    if self.loss_type == "hr":
                        loss = super_res_loss_s
                    else:#lr+hr
                        losses = [loss_s, super_res_loss_s]
                        idx = np.argmax([loss_s.item(), super_res_loss_s.item()])
                        loss = losses[idx]
                    #loss = torch.mean(torch.tensor([recons_loss_s.item(), super_res_loss_s.item()]))
                    #loss.requires_grad_()
                else:
                    super_res_loss_s = None
                    loss = loss_s
                    
                                
                if self.type == "vae":
                    loss = super_res_loss_s + kld_weight * kl
                else:
                    loss = super_res_loss_s
                    
            else:
                super_res_loss_s = None
                loss = loss_s
            
            
            if self.type == "vae":
                dict =  {'loss': loss, 'Reconstruction Loss Signal':recons_loss_s, 'KLD':kl.detach()}
            else:
                dict = {'loss': loss, 'Reconstruction Loss Signal':recons_loss_s}
            
            if self.sr_type == "convt":
                if "hr" in self.loss_type:
                    dict["Super Resolution Error"] = super_res_loss_s
                
            return dict

class VAE2d_SR_multimodal(nn.Module):

    """
    2 Dimensional VAE with SUPER RESOLUTION.
    The network is made up by stacking ENASMacroLayer. The Macro search space contains these layers.
    Each layer chooses an operation from predefined ones and SkipConnect then forms a network.

    Parameters
    ---
    num_layers: int
        The number of layers contained in the network.
    in_channel: int
        The number of input's channels.
    num_classes: int
        The number of classes for classification.
    dropout_rate: float
        Dropout layer's dropout rate before the final dense layer.
    ...
   """
    def __init__(self, to_predict, to_predict_hr = None, img = None, num_layers=3, out_filters=None, in_channels=12, kernel_sizes = None, strides = None, do = 0.2, batch_size = 1, loss = "mse",
                 scale_factor = None, size = [15, 250], activation_functions = None, width = 250, device = torch.device("cpu"), 
                 mode = "s", activation_name = "tanh", loss_name="mse", type = "ae", height = 15, sr_type = "upsample", loss_type = "lr", ks_sr = None, str_sr = None, of_sr = None,
                 ks_s_sr = None, str_s_sr = None, of_s_sr = None,  to_predict_s = None, to_predict_s_hr = None, out_filters_s = None, kernel_sizes_s = None, strides_s = None):
        """
        mode: 
            s - reconstruction error only signal
            w - reconstruction error only wavelets
            sw/ws - reconstruction error signal and wavelets
        """         
        
        super().__init__()

        self.supported_modes = ["w", "sw", "ws"]#no only signal
        self.supported_types = ["ae", "vae"]
        self.supported_sr_types = ["upsample", "convt", "none"]
        self.supported_loss_types = ["lr", "lr+hr", "hr"]
        
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.out_filters = out_filters
        self.in_channels = in_channels
        self.do = do 
        self.scale_factor = scale_factor
        self.kernel_sizes = kernel_sizes
        self.strides = strides 
        self.width = width 
        self.height = height
        self.to_predict = to_predict 
        self.to_predict_hr = to_predict_hr
        self.img = img 
        self.loss_name = loss_name
        self.nchs = in_channels 
    
        if loss_type not in self.supported_loss_types:
            loss_type = "lr"
        self.loss_type = loss_type 
        
        if sr_type == "none":
            sr_type = None
        elif sr_type not in self.supported_sr_types:
            sr_type = None
        self.sr_type = sr_type

        if mode not in self.supported_modes:
            raise Exception("Mode {} for Multimodal Super Resolution 2d VAE not supported. Supported modalities: 's' for only signal reconstruction error, 'w' for only wavelets reconstruction error and 'sw' for wavelets and signal mean reconstruction error".format(mode))
        else:
            if mode == "ws":
                mode = "sw"
        self.mode = mode 
            
        if "w" in self.mode:
            if self.to_predict_hr is None:
                raise Exception("to_predict_hr must be not None when ECG AE modality of reconstruction and super resolution is wavelet based.")
        
        if self.mode == "sw":
            self.ks_sr_s = ks_s_sr
            self.str_sr_s = str_s_sr
            self.of_sr_s = of_s_sr
            self.to_predict_s = to_predict_s
            self.to_predict_s_hr = to_predict_s_hr
            self.out_filters_s = out_filters_s
            self.kernel_sizes_s = kernel_sizes_s
            self.strides_s = strides_s
            self.input_shape_s = [in_channels, width]
            self.size_s = [width*10]
            
        if size is not None:
            self.size = size 
        else:
            self.size = width * 10
            if self.height is not None:
                self.size = [125, width*10]
                

        self.device = device
        self.decoder_input_shape = None
        self.z_input_shape = self.in_channels

        if self.out_filters is None:
            self.out_filters = [int(in_channels/(i+1)) for i in range(self.num_layers)]
            self.out_filters[-1] = 1
        if self.kernel_sizes is None:
            self.kernel_sizes = [[[1, 1], [1, 1]] for i in range(self.num_layers)]
        if self.strides is None:
            self.strides = [[[1, 1], [1, 1]] for i in range(self.num_layers)]
        
        ks_h = [elem[0] for items in self.kernel_sizes for elem in items]
        ks_w = [elem[1] for items in self.kernel_sizes for elem in items]
        str_h = [elem[0] for items in self.strides for elem in items]
        str_w = [elem[1] for items in self.strides for elem in items]
        sum_ksh = np.sum(ks_h)
        sum_strh = np.sum(str_h)
        sum_ksw = np.sum(ks_w)
        sum_strw = np.sum(str_w)
        sum_pad = len(ks_h)*2  
        self.x_input_shape = self.height - sum_ksh + sum_strh + sum_pad
        self.y_input_shape = self.width - sum_ksw + sum_strw + sum_pad
        self.latent_dim_nodense = (self.x_input_shape * self.y_input_shape)#self.num_layers * 2
        print("Latent dim Wavelets: {}".format(self.latent_dim_nodense))
        
        if "s" in self.mode:
            toprod =  [item for items in self.kernel_sizes_s for item in items]#item*2
            div = np.sum([elem for elem in toprod if elem >= 1]) 
            self.y_input_shape_s = int(self.width - div) 
            self.latent_dim_nodense_s = (1 * self.y_input_shape_s + len(self.kernel_sizes_s)*2)#self.num_layers * 2
            print("Latent dim ECG: {}".format(self.latent_dim_nodense_s))
            
        self.type = type.lower()
        if self.type not in self.supported_types:
            self.type = "ae"
        
        self.loss = loss
        self.activation_name = activation_name

        input_linear = self.latent_dim_nodense
        
        with torch.no_grad():
            
            torch.cuda.empty_cache()
            input_shape = (self.batch_size, self.in_channels, self.height, self.width) 
            
            if "s" in self.mode:
                #self.encoder = nn.Sequential(*modules).to(self.device)
                print("Encoder ks: ", self.kernel_sizes)
                input_shape_s = (self.batch_size, self.in_channels, self.width)
                self.encoder = Encoder1d(self.in_channels, self.out_filters_s, self.kernel_sizes_s, self.strides_s, self.do, self.activation_name, self.to_predict_s, self.img).to(self.device)
                #print(self.encoder)
                print("Encoder ECG: ", summary(self.encoder, input_shape_s), "\n")
                self.encoder_s_out =  self.latent_dim_nodense_s * self.batch_size * self.out_filters[-1][-1]
            if "w" in self.mode:
                #self.encoder_wav = nn.Sequential(*modules).to(self.device)
                self.encoder_wav = Encoder2d(self.in_channels, self.out_filters, self.kernel_sizes, self.strides, self.do, self.activation_name, self.to_predict, self.img).to(self.device)
                print("Encoder Wavelets: ", summary(self.encoder_wav, input_shape), "\n")
                self.encoder_out = self.latent_dim_nodense * self.batch_size * self.out_filters[-1][-1]
            
            if self.mode == "w":
                if self.type == "vae":
                    self.latent_dim = self.latent_dim_nodense*self.out_filters[-1][-1]
                    print("Mu/Var", self.latent_dim, self.latent_dim)
                    self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
                    self.fc_var = nn.Linear(self.latent_dim, self.latent_dim).to(self.device)
                else:
                    self.latent_dim = self.encoder_out
            else:
                if self.type == "vae":
                    raise Exception("vae not implemented for multimodality sr")
                else:
                    self.latent_dim = self.encoder_out
            torch.cuda.empty_cache()
       
        
        with torch.no_grad():
            
            torch.cuda.empty_cache()
            print("Width decoder: {}/{} = {}".format(self.latent_dim, self.out_filters[-1][-1],  self.x_input_shape, self.y_input_shape))
            self.decoder_input_shape = (self.batch_size, self.out_filters[-1][-1], self.x_input_shape, self.y_input_shape)
            print(self.decoder_input_shape)

            #self.decoder = nn.Sequential(*modules).to(self.device)
            self.ks_dec = reverse_listoflist(self.kernel_sizes)
            self.of_dec = reverse_listoflist(self.out_filters)
            self.str_dec = reverse_listoflist(self.strides)
            
            print("Decoder ks: ", self.ks_dec)
            print("Decoder stride: ", self.str_dec)
            if "s" in self.mode:
                self.ks_dec_s = reverse_listoflist(self.kernel_sizes_s)
                self.of_dec_s = reverse_listoflist(self.out_filters_s)
                self.str_dec_s = reverse_listoflist(self.strides_s)
                self.decoder_input_shape_s = (self.batch_size, self.out_filters_s[-1][-1], self.y_input_shape_s)
                self.decoder = Decoder1d(self.out_filters_s[-1][-1], self.of_dec_s, self.ks_dec_s, self.str_dec_s, self.do, self.activation_name, self.to_predict_s, self.img).to(self.device)
                print("Decoder ECG no SuperResolution: ", summary(self.decoder, self.decoder_input_shape_s), "\n")
            if "w" in self.mode:
                self.decoder_wav = Decoder2d(self.out_filters[-1][-1], self.of_dec, self.ks_dec, self.str_dec, self.do, self.activation_name, self.to_predict, self.img).to(self.device)
                print("Decoder GMW WAVELETS: ", summary(self.decoder_wav, self.decoder_input_shape), "\n")
                

            if self.sr_type == "upsample":
                if self.scale_factor is not None:
                    self.upsample = nn.Upsample(scale_factor = self.scale_factor, mode="bilinear")
                else:
                    self.upsample = nn.Upsample(size = self.size, mode="bilinear")
            elif self.sr_type == "convt":

                
                if of_sr is not None:
                    self.of_sr = of_sr
                else:
                    self.of_sr = self.of_dec
                if ks_sr is not None:
                    self.ks_sr = ks_sr
                else:
                    self.ks_sr = self.ks_dec
                if str_sr is not None:
                    self.str_sr = str_sr
                else:
                    self.str_sr = self.str_dec 
                        
                if self.nchs == 1:
                    if to_predict.dim() == 3:
                        to_predict = torch.squeeze(to_predict, 0)
                    if to_predict_hr.dim() == 3:
                        to_predict_hr = torch.squeeze(to_predict_hr, 0)
               
                print("Decoder SR ks: ", self.ks_sr)
                print("Decoder SR stride: ", self.str_sr)
                
                #do in the sr ? 
                if "w" in self.mode:
                    in_channels = self.out_filters[-1][-1]
                    self.decoder_sr_wav = Decoder2dSR(in_channels, self.of_sr, self.ks_sr, self.str_sr, self.do, self.activation_name, self.to_predict, self.img).to(self.device)
                    self.decoder_sr_input_shape = self.decoder_input_shape
                    print("Super Resolution ECG WAV decoder:", summary(self.decoder_sr_wav, self.decoder_sr_input_shape), "\n")
                if "s" in self.mode:
                    in_channels = self.out_filters_s[-1][-1]
                    #print(self.of_sr_s, self.ks_sr_s, self.str_sr_s)
                    self.decoder_sr_s = Decoder1dSR(in_channels, self.of_sr_s, self.ks_sr_s, self.str_sr_s, self.do, self.activation_name, self.to_predict_s, self.img).to(self.device)
                    self.decoder_sr_input_shape_s = self.decoder_input_shape_s
                    print("Super Resolution ECG WAV decoder:", summary(self.decoder_sr_s, self.decoder_sr_input_shape_s), "\n")
                
         
            if self.sr_type == "upsample":
          
                summary_decoder_sr = ''' \n ---------------------------------------------------- \n 
                                            |      Layer (type)      | |     Output Shape      |        
                                            ===================================================='''  
                for layer in range(self.num_layers):
                    if layer != 0:
                        i = layer+1
                    else:
                        i = layer+1

                    temp_w = [elem[1] for items in self.kernel_sizes for elem in items]
                    temp_h = [elem[0] for items in self.kernel_sizes for elem in items]
                    precw = self.width - np.sum(temp_w) #+ self.num_layers
                    prech = self.height - np.sum(temp_h)
                        
                    w = int(precw+temp_w[layer]) 
                    precw = w
                    h =  int(prech+temp_h[layer]) 
                    prech = h
                    ch = self.of_dec[layer] 
                    summary_decoder_sr += "   Conv2DTranspose-{}                    [-1, {}, {}, {}] \n".format(i, ch, h, w) 
                    #if layer != self.num_layers-1:
                    summary_decoder_sr += "    {}-{}                     [-1, {}, {}, {}] \n".format(activation_name.upper(), i, ch, h, w)
                        
                        
                if self.scale_factor is not None:
                    w = int(precw*self.scale_factor)
                    h = int(prech*self.scale_factor)
                else:
                    w = self.size[1]
                    h = self.size[0]
                
                summary_decoder_sr += "   Upsample-{}                  [-1, {}, {}, {}] \n".format(i, ch, h, w)
                print("Super resolution part of the Decoders: \n", summary_decoder_sr, "\n")
                
        torch.cuda.empty_cache()
            
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x H x W x C]
        :return: (Tensor) List of latent codes
        """
        torch.cuda.empty_cache()

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z_ecg = None
        z_wav = None
            
        if "w" in self.mode:
            
            if "s" in self.mode:
                x_s, x_s_hr, x_w, x_w_hr, scales, scales_hr, label, sublabel = x  #first is the signal
                z_ecg = self.encoder(x_s)
                z_ecg = torch.flatten(z_ecg)
                if z_ecg.ndim == 1:
                    z_ecg = torch.unsqueeze(z_ecg, dim = 0)
            else: 
                x_w, x_w_hr, scales, scales_hr, label, sublabel = x
        
            z_wav = self.encoder_wav(x_w)
            z_wav = torch.reshape(z_wav, (self.batch_size, self.latent_dim_nodense*self.out_filters[-1][-1]))
        
        if z_ecg is not None:
            if z_wav is not None:
                #print("z ecg: ", z_ecg.shape)
                #print("z wav: ", z_wav.shape)
                z = [z_ecg, z_wav] #testare
                z = torch.cat(z, dim = 1)
                #print("z final: ", z.shape)
            else:
                z = z_ecg
        else:
            if z_wav is not None:
                z = z_wav
        
        if self.type == "vae":
            mu = self.fc_mu(z)
            log_var = self.fc_var(z)
            return [mu, log_var]
        else:
            return z


    def decode(self, z, scales):
        """
        NO SUPER RESOLUTION DECODER 
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x H x W x C]
        """
        torch.cuda.empty_cache()
        
        recons_ecg = None
        recons_wav = None
        
        
        if "s" in self.mode:
            total = self.batch_size * self.out_filters_s[-1][-1] * self.latent_dim_nodense_s
            z_s = z[0, :total]
            if z_s.ndim == 1:
                z_s = torch.unsqueeze(z_s, dim=0)
            z_s = torch.reshape(z_s, (self.batch_size, self.out_filters_s[-1][-1], self.latent_dim_nodense_s))
            recons_ecg = self.decoder(z_s).to(self.device)           
            
            z_w = z[0, total:]
            if z_w.ndim == 1:
                z_w = torch.unsqueeze(z_w, dim=0)
            z_w = torch.reshape(z_w, (self.batch_size, self.out_filters[-1][-1], self.x_input_shape, self.y_input_shape))
        else:
            z_w = z
        
        if "w" in self.mode:
            if z_w.ndim == 1:
                z_w = torch.unsqueeze(z_w, dim=0)
            z_w = torch.reshape(z_w, (self.batch_size, self.out_filters[-1][-1], self.x_input_shape, self.y_input_shape))
            recons_wav = self.decoder_wav(z_w).to(self.device)
            if "w" == self.mode: #if sw, don't need to compute signal reconstruction from wavelets
                if scales is not None:
                    if scales.ndim == 2:
                        scale = scales[0]
                    else:
                        scale = scales
                    scale = torch.unsqueeze(scale, dim = 0)
                    scale = scale.cpu().detach().numpy()
                    
                                            
                    fs = 50
                    dt = 1/fs
                    dj = 0.5
                    #scale = np.arange(1, 15, dj)
                    recons_ecg = []
                    batches =  recons_wav.clone()
                        
                    for batch in batches:
                        signal = []
                        for ch in range(self.nchs):
                        
                            if batches.ndim == 2:
                                batches = torch.unsqueeze(batches, dim = 0)
                            if batches.ndim == 3:
                                batches = torch.unsqueeze(batches, dim = 0)

                            wav_ch = batch[ch].cpu().detach().numpy()
                            #print(wav_ch.shape, scale.shape)
                            ecg_ch = pycwt.icwt(wav_ch, scale, dt, dj = dj, wavelet='morlet')
                            signal.append(ecg_ch)
                        signal = torch.from_numpy(np.array(signal)).float().to(self.device)
                        recons_ecg.append(signal)
                    recons_ecg = torch.stack(recons_ecg).float().to(self.device)
                else:
                    recons_ecg = None
            
            return recons_wav, recons_ecg
        
        if recons_ecg is not None:
            if recons_wav is not None:
                result = [recons_wav, recons_ecg]
            else:
                result = recons_ecg
        else:
            if recons_wav is not None:
                result = recons_wav
        
        return result
    
    
    def decode_sr(self, z, scales):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H_HR x W_HR]
        """
        torch.cuda.empty_cache()
        
        result_ecg_sr = None
        result_wv_sr = None 
        if "s" in self.mode:
            total = self.batch_size * self.out_filters_s[-1][-1] * self.latent_dim_nodense_s
            z_s = z[0, :total]
            if z_s.ndim == 1:
                z_s = torch.unsqueeze(z_s, dim=0)
            z_s = torch.reshape(z_s, (self.batch_size, self.out_filters_s[-1][-1], self.latent_dim_nodense_s))
            recons_ecg = self.decoder(z_s).to(self.device)
            if self.sr_type == "upsample":
                result_ecg_sr = self.upsample(recons)
            elif self.sr_type == "convt":
                result_ecg_sr = self.decoder_sr_s(z_s).to(self.device)       
                
            z_w = z[0, total:]
            if z_w.ndim == 1:
                z_w = torch.unsqueeze(z_w, dim=0)
            z_w = torch.reshape(z_w, (self.batch_size, self.out_filters[-1][-1], self.x_input_shape, self.y_input_shape))
        else:
            z_w = z
            
        if "w" in self.mode:
            if z_w.ndim == 1:
                z_w = torch.unsqueeze(z_w, dim=0)
            z_w = torch.reshape(z_w, (self.batch_size, self.out_filters[-1][-1], self.x_input_shape, self.y_input_shape))
            recons = self.decoder_wav(z_w)
            if self.sr_type == "upsample":
                result_wv_sr = self.upsample(recons)
                if result_wv_sr.dim() == 2:
                    result_wv_sr = torch.unsqueeze(result_wv_sr, dim = 0)
                if result_wv_sr.dim() == 3:
                    result_wv_sr = torch.unsqueeze(result_wv_sr, dim = 1)
            elif self.sr_type == "convt":
                result_wv_sr = self.decoder_sr_wav(z_w).to(self.device)
                if result_wv_sr.dim() == 3:
                    result_wv_sr = torch.unsqueeze(result_wv_sr, dim = 1)
                    
            #print("sr", result_wv_sr.shape)
            if scales is not None:
                if scales.ndim == 2:
                    scale = scales[0]
                else:
                    scale = scales
                scale = torch.unsqueeze(scale, dim = 0)
                scale = scale.cpu().detach().numpy()
                
                
                result_ecg_sr = []
                fs = 500
                dt = 1/fs
                dj = 0.1
                #scale = np.arange(1, 15, dj)
                for batch in result_wv_sr:
                    signal = []
                    for ch in range(self.nchs):
                        if batch.dim() == 2:
                            batch = torch.unsqueeze(batch, dim = 0)
                        elif batch.dim() == 4:
                            batch = torch.squeeze(batch, dim = 1)
                        wav_ch = batch[ch].cpu().detach().numpy()
                        #print(wav_ch.shape, scale.shape)
                        ecg_ch = pycwt.icwt(wav_ch, scale, dt, dj = dj, wavelet='morlet')
                        signal.append(ecg_ch)
                    signal = torch.from_numpy(np.array(signal)).float().to(self.device)
                    result_ecg_sr.append(signal)                
                result_ecg_sr = torch.stack(result_ecg_sr).float().to(self.device)
                #print(result_ecg_sr.shape)
            else:
                result_ecg_sr = None
                
        #if "s" == self.mode:
        #    return result_ecg_sr
        if "w" in self.mode: #"w" and "sw"
            return result_wv_sr, result_ecg_sr
 

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    
    def kl_divergence(self, z, mu, log_var):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        std = torch.exp(0.5 * log_var)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, x):
        
        original_x = x.copy()
        
        if "s" == self.mode:

            if "hr" in self.loss_type:
                signal, signal_hr, label, sublabel = x
                x_return = [signal, signal_hr]
            else:
                signal, label, sublabel = x
                x_return = signal

        elif "w" == self.mode:
            
            wav, wav_hr, scales, scales_hr, label, sublabel = x
            
            #if wav.dim() > 3:
            #    wav = torch.squeeze(wav, dim = 1)
            #if wav_hr is not None:
            #    if wav_hr.dim() > 3:
            #        wav_hr = torch.squeeze(wav_hr, dim = 1)
            if scales is not None:
                if scales.ndim == 1:
                    scales = torch.unsqueeze(scales, dim = 0)
            if scales_hr is not None:
                if scales_hr.ndim == 1:
                    scales_hr = torch.unsqueeze(scales_hr, dim = 0)
                #print("Input", wav.shape, wav_hr.shape)
            #else:
                #print("Input", wav.shape, None)
                
            if scales is not None:
                dj = 0.5
                fs = 50
                dt = 1/fs
                if scales.ndim == 2:
                    scale = scales[0]
                else:
                    scale = scales
                scale = torch.unsqueeze(scale, dim = 0)
                scale = scale.cpu().detach().numpy()
                #scale = np.arange(1, 15, dj)
                signals = []
                for batch in wav:
                    signal = []
                    for ch in range(self.nchs):
                        wav_ch = batch[ch].cpu().detach().numpy()
                        #print(wav_ch.shape, scale.shape)
                        ecg = pycwt.icwt(wav_ch, scale, dt, dj = dj, wavelet='morlet')
                        signal.append(ecg)
                    #print(ch, ecg.shape)
                    signal = torch.from_numpy(np.array(signal)).float()
                    signals.append(signal)
                signal_lr = torch.stack(signals)
                signal_lr = signal_lr.requires_grad_()
            else:
                signal_lr = None
                
            
            if scales_hr is not None:
                dj = 0.1
                fs = 500
                dt = 1/fs
                if scales_hr.ndim == 2:
                    scale = scales_hr[0]
                else:
                    scale = scales_hr
                scale = torch.unsqueeze(scale, dim = 0)
                scale = scale.cpu().detach().numpy()
                #scale = np.arange(1, 15, dj)
                
                signals = []
                for batch in wav_hr:
                    signal = []
                    for ch in range(self.nchs):
                        wav_ch = batch[ch].cpu().detach().numpy()
                        #print(wav_ch.shape, scale.shape)
                        ecg = pycwt.icwt(wav_ch, scale, dt, dj, wavelet='morlet')
                        signal.append(ecg)
                    #print(ch, ecg.shape)
                    signal = torch.from_numpy(np.array(signal)).float()
                    signals.append(signal)
                signal_hr = torch.stack(signals)
                signal_hr = signal_hr.requires_grad_()
            else:
                signal_hr = None
            
            x_return = [signal_lr, signal_hr, wav, wav_hr] 
            #print(signal.shape)
            
        elif self.mode == "sw":

            signal, signal_hr, wav, wav_hr, scales, scales_hr, label, sublabel = x
            x_return = [signal, signal_hr, wav, wav_hr] 
           
        if self.type == "vae":
            mu, log_var = self.encode(x) #for VAE
            z = self.reparameterize(mu, log_var)
        else:
            z = self.encode(x)
        
        #print("Encoded ", z.shape)
        
        if z.ndim == 1:
            z = torch.unsqueeze(z, dim=0)
            #print("z after unsqueeze 0 dim", z.shape)
        
        #print("Input Decoders ", z.shape)
        
            
        if "w" in self.mode:
            #print("trying reshape", self.batch_size, self.out_filters[-1][-1], self.x_input_shape, self.y_input_shape)
            recons_wav, recons_ecg = self.decode(z, scales)
            recons_wav = torch.squeeze(recons_wav, dim = 0)
            if recons_ecg is not None:
                recons_ecg = torch.squeeze(recons_ecg, dim = 0)                
            
            #print("Recons WAV", recons_wav.shape)
            #print("Recons ECG", recons_ecg.shape)
            if self.sr_type is not None:
                recons_wav_sr, recons_ecg_sr = self.decode_sr(z, scales_hr)
                recons_wav_sr = torch.squeeze(recons_wav_sr, dim = 0)
                if recons_ecg_sr is not None:
                    recons_ecg_sr = torch.squeeze(recons_ecg_sr, dim = 0)
            else: 
                recons_wav_sr = None
                recons_ecg_sr = None 
            #print("Recons WAV SuperResolution ", recons_wav_sr.shape)    
            if self.sr_type is not None:
                if recons_ecg_sr  is not None:
                    #print("Recons ECG SuperResolution ", recons_ecg_sr.shape)
                    pass
                    
        if self.type == "vae":
            return  [[recons_ecg, recons_wav], [recons_ecg_sr, recons_wav_sr], x_return, mu, log_var] #vae
        else:
            return  [[recons_ecg, recons_wav], [recons_ecg_sr, recons_wav_sr], x_return, z]#ae
    
    
    
    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the Multimodal VAE with super resolution loss function.
        mse
        :param args:
        :param kwargs:
        :return:
        """
        torch.cuda.empty_cache()
        x = args[2]
        kld_weight = kwargs["kld_weight"]
            
        if self.mode == "s":
            
            recons_s = args[0]
            super_res_s = args[1]
            recons_s = torch.unsqueeze(recons_s, dim=0)

        else:
            
            x_s, x_s_hr, x_w, x_w_hr = x
    
            recons_s, recons_w = args[0]
            recons_s = torch.unsqueeze(recons_s, dim=0)
            recons_w = torch.unsqueeze(recons_w, dim=0)
            super_res_s, super_res_w = args[1]

        if self.type == "vae":
            
            mu = args[3]
            log_var = args[4]

            #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
            #print(kld_weight, "KLD weight")
            #kld_loss = - 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())# torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 0), dim = 0)#torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
             
            #expectation under z of the kl divergence between q(z|x) and
            #a standard normal distribution of the same shape
            z = self.reparameterize(mu, log_var)
            kl = self.kl_divergence(z, mu, log_var)
            
        loss_f = get_loss_function(self.loss_name) 

        if "s" in self.mode:
            recons_loss_s = loss_f(recons_s, x_s)
            if self.sr_type == "convt":
                if "hr" in self.loss_type:

                    super_res_loss_s = loss_f(super_res_s, x_s_hr)
                    losses = [recons_loss_s, super_res_loss_s]
                    idx = np.argmax([recons_loss_s.item(), super_res_loss_s.item()])
                    temp_loss_s = losses[idx]
                    #temp_loss_s = torch.mean(torch.tensor([recons_loss_s.item(), super_res_loss_s.item()]))
                    #temp_loss_s = torch.from_numpy(temp_loss_s)
                    #temp_loss_s.requires_grad_()
                else:
                    super_res_loss_s = None
                    temp_loss_s = recons_loss_s
            else:
                super_res_loss_s = None
                temp_loss_s = recons_loss_s
                
            if self.type == "vae":
                #loss_s = torch.mean(recons_loss_s + kld_weight * kld_loss)#.item()
                #elbo
                loss_s = (kl - temp_loss_s)
                loss_s = loss_s.mean()
            else:
                kl = None 
                loss_s = temp_loss_s
            
        if "w" in self.mode: #input mode wavelets
            recons_loss_w = loss_f(recons_w, x_w) #loss_f(recons_s.to(self.device), x_s.to(self.device)) It's better to learn the super resolution that optimize the signal reconstruction and super resulution, even when in input we have wavelets
            if self.sr_type == "convt":
                if "hr" in self.loss_type:
                    super_res_loss_w = loss_f(super_res_w, x_w_hr) #loss_f(super_res_s.to(self.device), x_s_hr.to(self.device))
                    if "lr" in self.loss_type:
                        losses = [recons_loss_w, super_res_loss_w]
                        idx = np.argmax([recons_loss_w.item(), super_res_loss_w.item()])
                        temp_loss_w = losses[idx]
                        #temp_loss_w = torch.mean(torch.tensor([recons_loss_w.item(), super_res_loss_w.item()]))
                        #temp_loss_w.requires_grad_()
                    else: 
                        recons_loss_w = None
                        temp_loss_w = super_res_loss_w
                else:
                    super_res_loss_w = None
                    temp_loss_w = recons_loss_w
            else:
                super_res_loss_w = None
                temp_loss_w = recons_loss_w
            
            if self.type == "vae":
                #loss_w = torch.mean(recons_loss_w + kld_weight * kld_loss)#.item()
                #elbo
                loss_w = (kl - temp_loss_w)
                loss_w = loss_w.mean()
            else:
                kl = None
                loss_w = temp_loss_w
            
        if self.mode == "sw": #MAYBE ERROR
            loss = torch.mean(torch.tensor([loss_s.item(), loss_w.item()]))
            loss.requires_grad_()
            return self.make_lossdict(loss, recons_loss_s=recons_loss_s, recons_loss_w=recons_loss_w, super_res_loss_s=super_res_loss_s, super_res_loss_w=super_res_loss_w, kl = kl)
        elif self.mode == "s":
            loss = loss_s
            return self.make_lossdict(loss, recons_loss_s=recons_loss_s, kl=kl)
        else:#self.mode == "w":
            loss = loss_w
            return self.make_lossdict(loss, recons_loss_w=recons_loss_w, super_res_loss_w=super_res_loss_w, kl=kl)

    def make_lossdict(self, loss, recons_loss_s = None, recons_loss_w = None, super_res_loss_s = None, super_res_loss_w = None, kl = None):
         
        loss_dict = {}
        loss_dict["loss"] = loss
        if self.type == "vae":
            loss_dict["KLD"] = kl.detach()
        if "s" in self.mode:
            loss_dict["Reconstruction Loss Signal"] = recons_loss_s
            if self.sr_type == "convt":
                if "hr" in self.loss_type:
                    loss_dict["Super Resolution Error Signal"] = super_res_loss_s
        if "w" in self.mode:
            if "lr" in self.loss_type:
                loss_dict["Reconstruction Loss Wavelets"] = recons_loss_w
            if self.sr_type == "convt":
                if "hr" in self.loss_type:
                    loss_dict["Super Resolution Error Wavelets"] = super_res_loss_w

        return loss_dict

def listoftensors2tensor(listoftensors):
    
    n_tensors = len(listoftensors)
    shape_ref = listoftensors[0].shape
    same_shape = True
    
    for tensor in listoftensors:
        if tensor.shape != shape_ref:
            same_shape = False
            break 
    if not same_shape:
        raise Exception("Cannot convert a list of tensors in a pytorch tensor object because shapes of the tensors in the list are not the same, {} != {}".format(shape_ref, tensor.shape))

    shapetensor = [n_tensors]
    for dim in shape_ref:
        shapetensor.append(dim)
    #gc.collect()
    torch.cuda.empty_cache()
    finaltensor = torch.zeros(shapetensor).to(device)

    for i, tensor in enumerate(listoftensors):
        #gc.collect()
        print(i, "/", len(listoftensors), end = "\r")
        #print(finaltensor.shape, tensor.shape)
        finaltensor[i] = tensor.to(device)
        del tensor
    return finaltensor            

def plot_prediction(model, data, label, fig, axs, nchs, scale = None, channel = None, loss = None, i=None, current_epoch=None, mode = "s", loss_name = "mse", batch_size = 1, device = "cuda:0", fig2 = None, axs2 = None, to_predict_hr = None):
    
    data = data.to(device)
    
    print(data.shape)
    
    n = data.shape[-1]
    t = np.arange(0, n, 1)
    model.eval()
    if mode == "w":
        if scale is None:
            raise Exception("scale must be not None if mode = 'w'")
            if channel is not None:
                scale = scale[channel]
            #dj = 0.1
            #scale = np.arange(1, 15, dj)
        if to_predict_hr is not None:
            data = [data, to_predict_hr, scale, None, None, None]#hr none for prediction
        else:
            data = [data, None, scale, None, None, None]#hr none for prediction
        pred = model(data)
        
    elif mode == "s":
        if to_predict_hr is not None:
            data = [data, to_predict_hr, None, None]
        else:
            data = [data, None, None, None]
        pred = model(data)
    model.train()
    
    
    #multi = False
    #if isinstance(pred[0], list):
    #    multi = True
        
    #if not multi:
    #    sig_recons = pred[0]#torch.unsqueeze(pred[0], dim=0)
    #    sig_or = pred[2]#torch.unsqueeze(pred[2], dim=0)
        
    #else:
    
    #0 recons 1 super res  2 original 3 latent
    #0 signal 1 wav lr 2 wav lr
    #0, 0 reconstructed ecg signal 
    #2, 0 original ecg signal
    
    #print(sig_or)
    #print(sig_recons)
    
    if mode == "s":
        sig_recons = pred[0]
        sig_or = pred[2]
        sig_sr = pred[1]
        if isinstance(sig_or, list):
            sig_or, sig_or_hr = sig_or
            hr = True 
        else:
            hr = False
        if nchs == 1:
            sig_or = torch.squeeze(sig_or, dim = 0)
            if hr:
                sig_or_hr = torch.squeeze(sig_or_hr, dim = 0)
                sig_or_hr = sig_or_hr.to(device)
            sig_recons = torch.squeeze(sig_recons, dim = 0)
            sig_sr = torch.squeeze(sig_sr, dim = 0)
        if hr:
            if sig_or_hr.ndim == 3:
                sig_or_hr = torch.squeeze(sig_or_hr, dim = 0)
    
    elif mode == "w":
        sig_recons = pred[0][0] 
        sig_or = pred[2][0] 
        sig_sr = pred[1][0]
        if isinstance(sig_or, list):
            sig_or, sig_or_hr = sig_or
            hr = True 
        else:
            hr = False
        if nchs == 1:
            sig_or = torch.squeeze(sig_or, dim = 0)
            #sig_sr = torch.squeeze(sig_sr, dim = 0)
            #if hr:
                #sig_or_hr = torch.squeeze(sig_or_hr, dim = 0)
                #sig_or_hr = sig_or_hr.to(device)
            print(sig_or.shape)
            #sig_recons = torch.unsqueeze(sig_recons, dim = 0)

    sig_or = sig_or.to(device)
    sig_recons = sig_recons.to(device)
    if sig_sr is not None:
        sig_sr = sig_sr.to(device)
        
    if sig_or.ndim == 3:
        sig_or = torch.squeeze(sig_or, dim = 0)
    print(sig_or.shape, sig_recons.shape)
    
    loss_f = get_loss_function(loss_name)
    recons_loss = loss_f(sig_or, sig_recons).item()#device
    print("Reconstruction error: {}".format(recons_loss))  

    if sig_sr is not None:
        
        loss_fsr = get_loss_function(loss_name)
        sig_sr = sig_sr.cpu().detach().numpy()
        sig_or_hr = sig_or_hr.cpu().detach().numpy()
        if sig_or_hr.ndim == 3:
            sig_or_hr = torch.squeeze(sig_or_hr, dim = 0)
        if not torch.is_tensor(sig_or_hr):
            if not isinstance(sig_or_hr, np.ndarray):
                sig_or_hr = np.array(sig_or_hr)
            sig_or_hr = torch.from_numpy(sig_or_hr)
        if not torch.is_tensor(sig_sr):
            if not isinstance(sig_sr, np.ndarray):
                sig_sr = np.array(sig_sr)
            sig_sr = torch.from_numpy(sig_sr)
        print(sig_or_hr.shape, sig_sr.shape)
        sr_loss = loss_fsr(sig_or_hr, sig_sr).item()
        print("Super resolution error: {}".format(sr_loss))
    
    if loss == None:
        loss = recons_loss
    
    sig_recons = sig_recons.cpu().detach().numpy()
    sig_or = sig_or.cpu().detach().numpy()
        
    if batch_size > 1:
        sig_or = sig_or[0, :, :]
        sig_recons = sig_recons[0, :, :]
        if sig_sr is not None:
            sig_sr = sig_sr[0, :, :]
            sig_or_hr = sig_or_hr[0, :, :]
        
    """
    if i is not None:
        print("Patient {}, label {}".format(i, label))
    """
    
    if current_epoch is not None:
        print("Epoch {}, label {}".format(current_epoch, label))    
    
    if nchs > 1:
        
        fig.subplots_adjust(wspace=0, hspace=1)
        #fig.tight_layout(pad=0.000)
        
        for j in range(nchs):
            
            y_min = min([min(sig_or[j, :]), min(sig_recons[j, :])])
            y_max = max([max(sig_or[j, :]), max(sig_recons[j, :])])
            
            for line in axs[j].get_lines(): # ax.lines:
                line.remove()
            axs[j].plot(t, sig_or[j, :], "g")
            axs[j].plot(t, sig_recons[j, :], "r--")
            axs[j].title.set_text("Channel {}".format(j+1))
            if j == 0:
                axs[0].set_title('Epoch {}, Loss: {}'.format(current_epoch, recons_loss))
            
            axs[j].set_ylim([y_min, y_max])
    else:
        
        print(sig_or.shape, sig_recons.shape)
        y_min = min([min(sig_or[:]), min(sig_recons[:])])
        y_max = max([max(sig_or[:]), max(sig_recons[:])])
        
        for line in axs.get_lines(): # ax.lines:
            line.remove()
        axs.plot(t, sig_or[:], "g")
        axs.plot(t, sig_recons[:], "r--")
        axs.set_ylim([y_min, y_max])
        axs.set_title('Epoch {}, Channel: {}, Loss: {}'.format(current_epoch, channel, recons_loss))
    # updating data values
    # line1.set_xdata(x)
    # line1.set_ydata(new_y)

    # drawing updated values
    fig.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig.canvas.flush_events()

    if mode == "s":
        
        if sig_sr is not None:
            if fig2 is not None:
                if axs2 is not None:

                    if nchs>1:
                        if sig_sr.ndim == 3:
                            sig_sr = torch.squeeze(sig_sr, dim=0)

                        if sig_or_hr.ndim == 3:
                            sig_or_hr = torch.squeeze(sig_or_hr, dim = 0)

                    sig_sr = sig_sr.cpu().detach().numpy()
                    sig_or_hr = sig_or_hr.cpu().detach().numpy()
                    n = sig_or_hr.shape[-1]
                    t = np.arange(0, n, 1)

                    if nchs > 1:

                        fig2.subplots_adjust(wspace=0, hspace=1)
                        #fig2.tight_layout(pad=0.000)
                        for j in range(nchs):


                            y_min = min([min(sig_or_hr[j, :]), min(sig_sr[j, :])])
                            y_max = max([max(sig_or_hr[j, :]), max(sig_sr[j, :])])

                            for line in axs2[j].get_lines(): # ax.lines:
                                line.remove()
                            axs2[j].plot(t, sig_or_hr[j, :], "g")
                            axs2[j].plot(t, sig_sr[j, :], "r--")
                            axs2[j].title.set_text("Channel {}".format(j+1))
                            if j == 0:
                                axs2[0].set_title('Epoch {}, Loss: {}'.format(current_epoch, sr_loss))

                            axs2[j].set_ylim([y_min, y_max])
                    else:

                        sig_or_hr = sig_or_hr.flatten()
                        sig_sr = sig_sr.flatten()
                        print(sig_or_hr.shape, sig_sr.shape)
                        y_min = min([min(sig_or_hr[:]), min(sig_sr[:])])
                        y_max = max([max(sig_or_hr[:]), max(sig_sr[:])])

                        for line in axs2.get_lines(): # ax.lines:
                            line.remove()
                        axs2.plot(t, sig_or_hr[:], "g")
                        axs2.plot(t, sig_sr[:], "r--")
                        axs2.set_ylim([y_min, y_max])
                        axs2.set_title('Epoch {}, Channel: {}, Loss: {}'.format(current_epoch, channel, sr_loss))
                    # updating data values
                    # line1.set_xdata(x)
                    # line1.set_ydata(new_y)

                    # drawing updated values
                    fig2.canvas.draw()

                    # This will run the GUI event
                    # loop until all UI events
                    # currently waiting have been processed
                    fig2.canvas.flush_events()
    
    del data
        
def plot_super_resolution(model, to_predict, to_predict_hr, ch, fig, axs, nchs, loss, current_epoch, scale, scale_hr, mode = "s", batch_size = 1, device = "cuda:0", loss_name = "mse"):
    
    
    to_predict = to_predict.to(device)
    model.eval()
    if mode == "w":
        data = [to_predict, to_predict_hr, scale, scale_hr, None, None]
        pred = model(data)
    elif mode == "s":
        data = [to_predict, to_predict_hr, None, None]
        pred = model(data)
    model.train()

    print(len(pred))
        
    n1 = to_predict.shape[-1]
    t1 = np.arange(0, n1, 1)

    #multi = False
    #if isinstance(pred[0], list):
    #    multi = True
        
    #if not multi:
    #    sig_recons = pred[0]#torch.unsqueeze(pred[0], dim=0)
    #    sig_sr = pred[1]
    #    sig_or = pred[2]#torch.unsqueeze(pred[2], dim=0)
    #else: 
    #    sig_recons = pred[0][0] #0 signal 1 wav
    #    sig_sr = pred[1][0]
    #    sig_or = pred[2][0] 
        
    if mode == "w":
        #0 recons, 1 super resolution, 2 input [lr, hr]
        #0 signal 1 wav lr 2 wav hr
        sig_recons = pred[0][0] 
        sig_sr = pred[1][0]
        sig_or = pred[2][0]
        sig_ecg_hr = pred[2][1].to(device)
        sig_ecg_sr = pred[1][0].to(device)
        
    elif mode == "s":
        sig_recons = pred[0] #0 recons, 1 super resolution, 2 input lr 
        sig_sr = pred[1]
        sig_or = pred[2]
        sig_ecg_hr = to_predict_hr.to(device)
        sig_ecg_sr = sig_sr.to(device)
        if isinstance(sig_or, list):
            sig_or = sig_or[0]
    
            
    if batch_size > 1:
        sig_or = sig_or[0, :, :]
        sig_recons = sig_recons[0, :, :]
        
        print(sig_or.shape, sig_recons.shape, sig_ecg_hr.shape, sig_sr.shape, sig_ecg_sr.shape)
        if sig_sr is not None:
            sig_sr = sig_sr[0, :, :]
            sig_ecg_hr = sig_ecg_hr[0, :, :]
            sig_ecg_sr = sig_ecg_sr[0, :, :]
    
    loss_f = get_loss_function(loss_name)
    sr_loss = loss_f(sig_ecg_hr, sig_ecg_sr).item()
    print("Super resolution error: {}".format(sr_loss))
    
    if nchs == 1:
        if sig_or.dim() == 2:
            sig_or = torch.squeeze(sig_or, dim = 0)
        if sig_recons.dim() == 2:
            sig_recons = torch.squeeze(sig_recons, dim = 0)
        if sig_sr.dim() == 2:
            sig_sr = torch.squeeze(sig_sr, dim = 0)
    else:
        if sig_or.ndim == 3:
             sig_or = torch.squeeze(sig_or, dim = 0)
        if sig_recons.dim() == 3:
            sig_recons = torch.squeeze(sig_recons, dim = 0)
        if sig_sr.dim() == 3:
            sig_sr = torch.squeeze(sig_sr, dim = 0)      
    n2 = sig_sr.shape[-1]
    t2 = np.arange(0, n2, 1)
    fig.suptitle("Super resoluted ECG signal, channel {}. Loss: {}".format(ch +1, sr_loss))
    
    for i in range(3):
        for line in axs[i].get_lines(): # ax.lines:
            line.remove()
        if i == 2:
            t = t2
            title = "Super Resolution"
            if nchs > 1:
                x = sig_sr[ch, :].cpu().detach().numpy()
            else:
                x = sig_sr[:].cpu().detach().numpy()
                
            color = "orange"
            axs[i].plot(t, x, color)
            y_min = min(x)
            y_max = max(x)
            axs[i].set_ylim([y_min, y_max])
        else: 
            t = t1
            if i == 0:
                title = "Original Signal"
                if nchs > 1:
                    x = sig_or[ch, :].cpu().detach().numpy()
                else:
                    x = sig_or[:].cpu().detach().numpy()
                color = "g"
                
                axs[i].plot(t, x, color)
                
            else:
                title = "Reconstructed Signal without super resolution"
                y_mins = []
                y_maxs = []
                if nchs > 1:
                    x = sig_or[ch, :].cpu().detach().numpy()
                else:
                    x = sig_or[:].cpu().detach().numpy()
                color = "g"
                axs[i].plot(t, x, color)
                y_mins.append(min(x))
                y_maxs.append(max(x))
                    
                if nchs > 1:
                    x = sig_recons[ch, :].cpu().detach().numpy()
                else:
                    x = sig_recons[:].cpu().detach().numpy()
                y_mins.append(min(x))
                y_maxs.append(max(x))
                
                color = "r--"
                axs[i].plot(t, x, color)
                
                y_min = min(y_mins)
                y_max = max(y_maxs)
                axs[i].set_ylim([y_min, y_max])
        axs[i].title.set_text(title)
        
           # updating data values
           # line1.set_xdata(x)
           # line1.set_ydata(new_y)
        
    axs[0].set_title('Epoch {}'.format(current_epoch))
    # drawing updated values
    fig.canvas.draw()    
    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig.canvas.flush_events()
    
    del pred 
    del data 
    del to_predict
    del sig_ecg_sr
    del sig_ecg_hr
    del sr_loss
        
class MMVAESR_Experiment(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params, 
                 mode,
                 to_predict, 
                 label,
                 nchs,
                 batch_size,
                 channel = None,
                 log_epochs = 1,
                 epochs_save_preliminar = None,
                 lr_change = None,
                 loss_name = "mse",
                 lr_scheduler = False,
                 in_colab = False,
                 loss_type = "lr+hr",
                 to_predict_hr = None,
                 sr_type = None,
                 denoising = True
                ):
        
        super().__init__()
        self.params = params
        self.curr_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = vae_model.to(self.curr_device)
        self.mode = mode
        self.hold_graph = False
        self.log_epochs = log_epochs
        self.epochs_save_preliminar = epochs_save_preliminar
        self.to_predict = to_predict
        self.channel = channel 
        self.denoising = denoising
        self.in_colab = in_colab
        self.automatic_optimization = False
        if self.to_predict.ndim == 3:
            if self.to_predict.shape[0] == 1:
                self.to_predict = torch.squeeze(self.to_predict, dim = 0)
                
        self.label = label
        self.nchs = nchs
        self.batch_size = batch_size
        self.optimizer  = None
        self.lr_change = lr_change
        self.lr_scheduler = lr_scheduler
        self.fig_sr, self.ax_sr = plt.subplots(3, figsize=(10, 8))
        self.loss_name = loss_name
        self.loss_type = loss_type 
        self.to_predict_hr = to_predict_hr
        self.sr_type = sr_type 
        
        self.losses = {}
        
        if "hr" in self.loss_type:
            self.hr = True
        else:
            self.hr = False 
            
        self.supported_loss_types = ["lr", "lr+hr", "hr"]
        if loss_type not in self.supported_loss_types:
            loss_type = "lr"
        self.loss_type = loss_type 
        
        self.supported_sr_types = ["upsample", "convt", "none", None]
        if sr_type == "none":
            sr_type = None
        elif sr_type not in self.supported_sr_types:
            sr_type = None
        self.sr_type = sr_type

        
        if self.nchs > 1:
            print(">1", self.nchs)
            self.fig, self.ax = plt.subplots(self.nchs, figsize=(10, 8))
            if self.hr:
                self.fig2, self.ax2 = plt.subplots(self.nchs, figsize=(10, 8))
            else:
                self.fig2 = None
                self.ax2 = None
        else:
            print("ELSE", self.nchs)
            self.fig, self.ax = plt.subplots(1, figsize=(10, 8))
            if self.hr:
                self.fig2, self.ax2 = plt.subplots(1, figsize=(10, 8))
            else:
                self.fig2 = None
                self.ax2 = None
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
    
    def forward(self, x, **kwargs):
            
        #print([temp.shape for temp in x])
        return self.model(x, **kwargs)
    """
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
                    
        results = self.forward(batch)
        #print(results.shape)
        train_loss = self.model.loss_function(*results,
                                                  kld_weight = self.params['kld_weight'], 
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx = batch_idx,
                                                 )
        #self.model.log_end_epoch(self) 
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    """
    def training_step(self, batch, batch_idx):
        
        """
        # Access the optimizer and scheduler
        optimizer = self.optimizers()
        if self.lr_scheduler:
            scheduler = self.lr_schedulers()
        else:
            scheduler = None

        # Compute loss
        #print(len(batch))
        results = self.forward(batch)
        loss = self.model.loss_function(*results, kld_weight = self.params['kld_weight'])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)

        # Perform manual optimization
        optimizer.zero_grad()
        loss["loss"].backward()
        optimizer.step()

        # Optionally, update the learning rate using the scheduler
        if scheduler is not None:
            scheduler.step()
        
        return loss['loss']
        """
        
        # Access the optimizer and scheduler
        if self.sr_type is not None:
            optimizer_rec, optimizer_sr = self.optimizers()
        else:
            optimizer_rec = self.optimizers()

        
        if self.lr_scheduler:
            if self.sr_type is not None:
                scheduler_rec, scheduler_sr = self.lr_schedulers()
            else:
                scheduler_rec = self.lr_schedulers()
        else:
            scheduler_rec = None
            scheduler_sr = None

        # Compute loss
        
        results = self.forward(batch)
        loss = self.model.loss_function(*results, kld_weight= self.params['kld_weight'])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)

        # Perform reconstruction manual optimization
        optimizer_rec.zero_grad()
        loss_rec = loss["Reconstruction Loss Signal"]
        loss_rec.backward(retain_graph=True)
        optimizer_rec.step()
        
        if self.sr_type is not None:
            # Perform super resolution manual optimization
            if "hr" in self.loss_type:
                optimizer_sr.zero_grad()
                loss_sr = loss["Super Resolution Error"]
                loss_sr.backward()
                optimizer_sr.step()

        # Optionally, update the learning rate using the scheduler
        if scheduler_rec is not None:
            scheduler_rec.step()
        if scheduler_sr is not None:
            scheduler_sr.step()
        
        return loss["loss"]
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        
        results = self.forward(batch)
        val_loss = self.model.loss_function(*results,
                                                kld_weight = self.params['kld_weight'],
                                                optimizer_idx = optimizer_idx,
                                                batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_train_epoch_end(self):
        
        loss = self.logger.log_end_epoch() 
           
        print(self.current_epoch, end="\r")    
        if self.lr_change is not None:
            if self.current_epoch in self.lr_change.keys():
                new_lr = self.lr_change[self.current_epoch]
                print("Epoch {}, changing learning rate from {} to {}".format(self.current_epoch, self.params['LR'], new_lr))
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=new_lr)
        
        if not self.in_colab:
            if self.current_epoch  % self.log_epochs == 0 or self.current_epoch % self.epochs_save_preliminar == 0:
                clear_output()

                plot_prediction(self.model, self.to_predict, self.label, self.fig, self.ax, self.nchs, None, loss = loss, channel = self.channel, mode = self.mode, current_epoch=self.current_epoch, loss_name=self.loss_name, batch_size=self.batch_size, fig2 = self.fig2, axs2 = self.ax2, to_predict_hr = self.to_predict_hr)
                if self.channel is not None:
                    ch = self.channel
                else:
                    ch = 0
                if self.sr_type is not None:
                    plot_super_resolution(self.model, self.to_predict, self.to_predict_hr, ch, self.fig_sr, self.ax_sr, self.nchs, loss, self.current_epoch, None, None, batch_size=self.batch_size)
                #fig_latentspace = plot_latent_space(self.model, self.train_data, self.train_labels)
                #fig_pca_ls = plot_pca_latent_space(self.model, self.train_data, self.train_labels)
                self.fig.show()
                self.fig_sr.show()

            if self.current_epoch % self.epochs_save_preliminar == 0:
                clear_output()

                plot_prediction(self.model, self.to_predict, self.label, self.fig, self.ax, self.nchs, None, loss = loss, channel = self.channel, mode = self.mode, current_epoch=self.current_epoch, loss_name=self.loss_name, batch_size=self.batch_size, fig2 = self.fig2, axs2 = self.ax2, to_predict_hr=self.to_predict_hr)
                if self.channel is not None:
                    ch = self.channel
                else:
                    ch = 0
                if self.sr_type is not None:
                    plot_super_resolution(self.model, self.to_predict, self.to_predict_hr, ch, self.fig_sr, self.ax_sr, self.nchs, loss, self.current_epoch, None, None, batch_size=self.batch_size)#channel 0
                #fig_latentspace = plot_latent_space(self.model, self.train_data, self.train_labels)
                #fig_pca_ls = plot_pca_latent_space(self.model, self.train_data, self.train_labels)
                self.fig.savefig('training_figures/reconstruction_{}epoch.png'.format(self.current_epoch), dpi=self.fig.dpi)
                self.fig_sr.savefig('training_figures/super_resolution_channel0_{}epoch.png'.format(self.current_epoch), dpi=self.fig_sr.dpi)
                #fig_latentspace.savefig('training_figures/latent_space_labels_{}epoch.png'.format(curr_epoch), dpi=fig_latentspace.dpi)
                #fig_pca_ls.savefig('training_figures/pca_latent_space_labels_{}epoch.png'.format(curr_epoch), dpi=fig_pca_ls.dpi)
                #plt.show()
        
        print("Epoch: {}, Loss: {}".format(self.current_epoch, loss), end=' \r')
    
    def configure_optimizers(self):
      
        """
        optimizer = torch.optim.Adam(self.model.parameters(),
                               lr=self.params['LR']
                               )
        if self.lr_scheduler:
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
            return [optimizer], [scheduler]
        else:
            return [optimizer]
        """
        optimizer_rec = torch.optim.AdamW(self.model.parameters(),
                               lr=self.params['LR']
                               )
        optims = [optimizer_rec]
        if self.sr_type is not None:
            optimizer_sr = torch.optim.AdamW(self.model.parameters(),
                                   lr=self.params['LR']
                                   )
            optims.append(optimizer_sr)
        if self.lr_scheduler:
            scheduler_rec = LinearLR(optimizer_rec, start_factor=1.0, end_factor=0.1, total_iters=30, verbose = True)
            schedulers = [scheduler_rec]
            if self.sr_type is not None:
                scheduler_sr = LinearLR(optimizer_sr, start_factor=1.0, end_factor=0.1, total_iters=30, verbose = True)
                schedulers.append(scheduler_sr)
            return optims, schedulers
        else:
            return optims 
        
def plot_wavelets_sr_rt(model, to_predict, to_predict_hr, fig_wvsr, axs_wvsr, nchs = 12, batch_size = 1, epoch = None, device = "cuda:0", fs_lr = 50, fs_hr = 500, fs_sr = 500, ch = 0):

    wavelet_lr = to_predict.to(device)
    wavelet_hr = to_predict_hr.to(device)
    model.eval()
    pred = model([wavelet_lr, wavelet_hr, None, None, None, None])
    model.train()
    wavelet_sr = pred[1][1]
    wavelet_rec = pred[0][1]
    
    if batch_size > 1:
        wavelet_sr = wavelet_sr[0, :, :, :]
        wavelet_rec = wavelet_rec[0, :, :, :]
        wavelet_lr = wavelet_lr[0, :, :, :]
        wavelet_hr = wavelet_hr[0, :, :, :]
    
    criterion = nn.MSELoss()
    loss_rec = criterion(wavelet_rec, wavelet_lr)
    loss_sr = criterion(wavelet_sr, wavelet_hr)
    
    temp = [wavelet_lr, wavelet_rec, wavelet_hr, wavelet_sr]
    wavelets = []
    for wave in temp:
        if wave.ndim == 4:
            wave = torch.squeeze(wave, dim = 0)
        elif wave.ndim == 2:
            wave = torch.unsqueeze(wave, dim = 0)
        wavelets.append(wave)
    
    if len(wavelets)!=4:
        raise Exception("wavelets must be an array-like object with lenght equals to 4: low res, low res reconstruction, high res and super res reconstruction. Array-like object with len {} given".format(len(wavelets)))
        
    titles = ["Low Resolution GMW wavelets channel {}".format(ch), "Low Resolution GMW wavelets channel {} Reconstruction".format(ch), "High Resolution GMW wavelets channel {}".format(ch), "Super Resolution GMW wavelets channel {}".format(ch)]
    fig_wvsr.suptitle("Super Resolution Wavelets epoch {}. Reconstruction Loss: {}, Super resolution Loss: {}".format(epoch, loss_rec.detach(), loss_sr.detach()))
    
    vmaxs = []
    imgs = []
    for j in range(4):
        wave = wavelets[j]
        #wave = torch.squeeze(wave, dim = 0)
        img = axs_wvsr[j].imshow(np.abs(wave[ch].cpu().detach().numpy()), aspect='auto', cmap='turbo')    
        axs_wvsr[j].set_title(titles[j])
    
        divider = make_axes_locatable(axs_wvsr[j])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig_wvsr.colorbar(img, cax=cax, orientation='vertical')
        
        # Assume colorbar was plotted last one plotted last
        cb = img.colorbar   
        vmaxs.append(cb.vmax)
        imgs.append(img)
        cb.remove()
    
    idx = np.argmax(vmaxs)
    divider = make_axes_locatable(axs_wvsr[-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig_wvsr.colorbar(imgs[idx], cax=cax, orientation='vertical')

    # drawing updated values
    fig_wvsr.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig_wvsr.canvas.flush_events()
    
    del wavelet_lr
    del wavelet_hr
    #plt.legend()
    #print("Mean Squared Error: %.3g" % np.mean(np.abs(datas[1] - datas[0])**2))


def plot_wavelets_rt(model, to_predict, to_predict_hr, scale, scale_hr, fig_wv, axs_wv, fig_ecgw, axs_ecgw, batch_size = 1, nchs = 12, epoch = None, device = "cuda:0", fs_lr = 50, fs_hr = 500, fs_sr = 500):

    wavelet_lr = to_predict.to(device)
    wavelet_hr = to_predict_hr.to(device)
    model.eval()
    pred = model([wavelet_lr, wavelet_hr, scale, scale_hr, None, None])
    model.train()
    wavelet_sr = pred[1][1]
    wavelet_rec = pred[0][1]

    if batch_size > 1:
        wavelet_sr = wavelet_sr[0, :, :, :]
        wavelet_rec = wavelet_rec[0, :, :, :]
        wavelet_lr = wavelet_lr[0, :, :, :]
        wavelet_hr = wavelet_hr[0, :, :, :]
        scale = scale[0]
        scale_hr = scale_hr[0]
        #scale = np.arange(1, 15, 0.5)
        #scale = np.arange(1, 15, 0.1)
    sr = False
    if wavelet_sr is not None: 
        sr = True
        temp = [wavelet_lr, wavelet_hr, wavelet_sr]
        titles = ["Low Resolution GMW wavelets channel {}", "High Resolution GMW wavelets channel {}", "Super Resolution GMW wavelets channel {}"]
        titles_data = ["High Resolution signal at {}Hz".format(fs_hr), "Super Resoluted {}Hz signal from low resolution {}Hz".format(fs_sr, fs_lr)]
        colors = ["g", "r--"]
        widths = [4, 2]
        labels = ["high resolution", "super resoluted from low resolution"]
    else:
        temp = [wavelet_lr, wavelet_rec]
        titles = ["Low Resolution GMW wavelets channel {}", "Low Resolution Reconstruction GMW wavelets channel {}"]
        titles_data = ["Low resolution signal at {}Hz".format(fs_hr), "Reconstructed Low resolution  Resoluted {}Hz signal from low resolution {}Hz".format(fs_sr, fs_lr)]
        colors = ["g", "r--"]
        widths = [4, 2]
        labels = ["low resolution", "low resolution reconstruction"]
    
    wavelets = []
    for wave in temp:
        if wave.dim() == 4:
            wave = torch.squeeze(wave, 0)
        if wave.dim() == 2:
            wave = torch.unsqueeze(wave, 0)
        wavelets.append(wave)
        
    scales = [scale, scale_hr, scale_hr]
    
    datas = []#ecg lr/ ecg hr, ecg rec / ecg sr
    datas_all = []#lr, hr, sr/rec
    
    fss = [50, 500, 500]

    for i, wave in enumerate(wavelets):
        fs = fss[i]
        scale = scales[i]
        if fs == 500:
            dj = 0.1
         
        else:
            dj = 0.5
        dt = 1/fs
        
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale)
            
        if scale.ndim == 2:
            scale = scale[0]
            scale = torch.unsqueeze(scale, dim = 0)
            
        elif scale.ndim == 1:
            scale = torch.unsqueeze(scale, dim = 0)            

        rec = []
        for ch in range(nchs):
            wave_ch = wave[ch].cpu().detach().numpy()
            if not isinstance(scale, np.ndarray):
                scale = scale.cpu().detach().numpy()
            #print(wave_ch.shape, scale.shape)
            rec.append(pycwt.icwt(wave_ch, scale, dt, dj = dj, wavelet="morlet"))
        rec = np.array(rec)
        rec = torch.from_numpy(rec).float()

        if sr:
            if i == 0:
                datas_all.append(rec)
            else:
                datas_all.append(rec)
                datas.append(rec)     
        else:
            datas_all.append(rec)
            datas.append(rec)     
      
    if len(datas) != 2:
        raise Exception("datas must be an array-like object with lenght equals to 2:  high res and super res (or low res and reconstruction). Array-like object with len {} given".format(len(datas)))
    if len(wavelets) not in [2, 3]:
        raise Exception("wavelets must be an array-like object with lenght equals to 3 or 2: low res, high res and super res (low res and reconstruction). Array-like object with len {} given".format(len(wavelets)))
        

    datas = [data.cpu() for data in datas]

    fig_wv.suptitle("Wavelets epoch ".format(epoch))
    fig_ecgw.suptitle("ECG epoch {}".format(epoch))
    
    for ch in range(nchs):
        if nchs != 1:
            ax_e = axs_ecgw[ch]
            ax_w = axs_wv[ch]
        else:
            ax_e = axs_ecgw
            ax_w = axs_wv

        ax_e.cla()            
        imgs = []
        vmaxs = []
        for j in range(len(wavelets)):
            wave = wavelets[j]
            img = ax_w[j].imshow(np.abs(wave[ch].cpu().detach().numpy()), aspect='auto', cmap='turbo')    
            ax_w[j].set_title(titles[j].format(ch))
            divider = make_axes_locatable(ax_w[j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig_wv.colorbar(img, cax=cax, orientation='vertical')
    
            # Assume colorbar was plotted last one plotted last
            cb = img.colorbar   
            vmaxs.append(cb.vmax)
            imgs.append(img)
            cb.remove()
    
        idx = np.argmax(vmaxs)
        divider = make_axes_locatable(ax_w[-1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig_wv.colorbar(imgs[idx], cax=cax, orientation='vertical')
        
        for j in range(2):
            if nchs != 1:
                ax_e.plot(datas[j][ch, :].cpu(), colors[j], label = labels[j], linewidth = widths[j])
            else:
                ax_e.plot(datas[j][0, :].cpu(), colors[j], label = labels[j], linewidth = widths[j])
            # drawing updated values
    fig_wv.canvas.draw()
    fig_ecgw.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    fig_ecgw.canvas.flush_events()
    fig_wv.canvas.flush_events()
    
    del wavelet_lr
    del wavelet_hr
    
    #plt.legend()
    #print("Mean Squared Error: %.3g" % np.mean(np.abs(datas[1] - datas[0])**2))

class MMVAESR_Experiment2d(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params, 
                 mode,
                 to_predict,
                 label,
                 nchs,
                 batch_size,
                 to_predict_signal,
                 channel = None,
                 to_predict_hr = None,
                 log_epochs = 10,
                 epochs_save_preliminar = 100,
                 loss_name = "mse",
                 lr_change = None,
                 lr_scheduler = False,
                 in_colab = False,
                 sr_type = "upsample", 
                 loss_type = "lr", 
                 scales = None, 
                 scales_hr = None,
                 to_predict_hr_signal = None
                 ):
        
        super().__init__()

        self.params = params
        self.curr_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = vae_model.to(self.curr_device)
        self.mode = mode
        self.hold_graph = False
        self.log_epochs = log_epochs
        self.epochs_save_preliminar = epochs_save_preliminar
        self.to_predict = to_predict
        self.to_predict_hr = to_predict_hr
        self.label = label
        self.nchs = nchs
        self.channel = channel
        self.batch_size = batch_size 
        self.loss_name = loss_name
        self.fig_sr, self.axs_sr = plt.subplots(3, figsize=(10, 8))
        self.in_colab = in_colab
        self.lr_scheduler = lr_scheduler
        self.automatic_optimization = False
        self.scales = scales
        self.scales_hr = scales_hr 
        
        self.to_predict_signal = to_predict_signal
        self.to_predict_hr_signal = to_predict_hr_signal
        
        self.supported_loss_types = ["lr", "lr+hr", "hr"]
        if loss_type not in self.supported_loss_types:
            loss_type = "lr"
        self.loss_type = loss_type 
        
        self.supported_sr_types = ["upsample", "convt", "none", None]
        if sr_type == "none":
            sr_type = None
        elif sr_type not in self.supported_sr_types:
            sr_type = None
        self.sr_type = sr_type

        if "w" in self.mode: 
            if self.to_predict_hr is None:
                raise Exception("to_predict_hr must be not None when ECG AE reconstruction and super resolution modality is wavelets based.")
            else:
                self.fig_wv, self.axs_wv = plt.subplots(self.nchs, 3, figsize=(10, 8))
                self.fig_wvsr, self.axs_wvsr = plt.subplots(1, 4, figsize=(10, 8))
                self.fig_ecgw, self.axs_ecgw = plt.subplots(self.nchs, 1, figsize=(10, 8))
                self.fig_recwav, self.axs_recwav = plt.subplots(1, 1, figsize=(10, 8))
                self.fig_recwav.suptitle("Signal reconstruction from reconstructed wavelet")
                self.fig_srwav, self.axs_srwav =  plt.subplots(1, 1, figsize=(10, 8))
                self.fig_srwav.suptitle("Signal Super-Resolution from super-resoluted wavelet")
        if self.nchs > 1:
            print(">1", self.nchs)
            self.fig, self.axs = plt.subplots(self.nchs, figsize=(10, 8))
        else:
            print("ELSE", self.nchs)
            self.fig, self.axs = plt.subplots(1, figsize=(10, 8))
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
    
    def forward(self, x, **kwargs):
        
        #if self.mode == "sw":
            #x = x[:2] #NO SCALES
        return self.model(x, **kwargs)
    
    """
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        #print(batch.shape)
        results = self.forward(batch)
        train_loss = self.model.loss_function(*results,
                                                  kld_weight = self.params['kld_weight'], 
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx = batch_idx,
                                                 )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    """
    def training_step(self, batch, batch_idx):
                                         
        # Access the optimizer and scheduler
        if self.sr_type is not None:
            optimizer_rec, optimizer_sr = self.optimizers()
        else:
            optimizer_rec = self.optimizers()

        
        if self.lr_scheduler:
            if self.sr_type is not None:
                scheduler_rec, scheduler_sr = self.lr_schedulers()
            else:
                scheduler_rec = self.lr_schedulers()
        else:
            scheduler_rec = None
            scheduler_sr = None

        # Compute loss
        
        results = self.forward(batch)
        loss = self.model.loss_function(*results, kld_weight = self.params['kld_weight'])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)

        # Perform reconstruction manual optimization
        if "lr" in self.loss_type:
            optimizer_rec.zero_grad()
            loss_rec = loss["Reconstruction Loss Wavelets"]
            loss_rec.backward(retain_graph=True)
            optimizer_rec.step()
        
        if self.sr_type is not None:
            # Perform super resolution manual optimization
            if "hr" in self.loss_type:
                optimizer_sr.zero_grad()
                loss_sr = loss["Super Resolution Error Wavelets"]
                loss_sr.backward(retain_graph=True) #no?
                optimizer_sr.step()

        # Optionally, update the learning rate using the scheduler
        if scheduler_rec is not None:
            scheduler_rec.step()
        if scheduler_sr is not None:
            scheduler_sr.step()
        
        return loss["loss"]
    
    def on_train_epoch_end(self):
        
        #print(self.curr_epoch)
        loss = self.logger.log_end_epoch()
        if not self.in_colab:
            if self.current_epoch % self.log_epochs == 0 or self.current_epoch % self.epochs_save_preliminar == 0:
                clear_output()
                plot_prediction(self.model, self.to_predict, self.label, self.fig, self.axs, self.nchs, self.scales, channel = self.channel, mode = self.mode, current_epoch=self.current_epoch, batch_size = self.batch_size)
                if self.channel is not None:
                    ch = self.channel
                else:
                    ch = 0
                if self.sr_type is not None:
                    plot_super_resolution(self.model, self.to_predict, self.to_predict_hr, ch, self.fig_sr, self.axs_sr, self.nchs, loss, self.current_epoch, self.scales, self.scales_hr, mode = self.mode, batch_size=self.batch_size)
                if "w" in self.mode:
                    plot_wavelet_reconstruction(self.to_predict_signal, self.to_predict, self.scales, self.fig_recwav, self.axs_recwav,  fs = 50)
                    
                    plot_wavelets_rt(self.model, self.to_predict, self.to_predict_hr, self.scales, self.scales_hr, self.fig_wv, self.axs_wv, self.fig_ecgw, self.axs_ecgw, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)        
                    if self.sr_type is not None:
                        plot_wavelet_reconstruction(self.to_predict_hr_signal, self.to_predict_hr, self.scales_hr, self.fig_srwav, self.axs_srwav,  fs = 500)
                        plot_wavelets_sr_rt(self.model, self.to_predict, self.to_predict_hr, self.fig_wvsr, self.axs_wvsr, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)        
                #fig_latentspace = plot_latent_space(self.model, self.train_data, self.train_labels)
                #fig_pca_ls = plot_pca_latent_space(self.model, self.train_data, self.train_labels)
                plt.show()

            if self.current_epoch % self.epochs_save_preliminar == 0:
                clear_output() 
                plot_prediction(self.model, self.to_predict, self.label, self.fig, self.axs, self.nchs, self.scales, channel = self.channel, mode = self.mode, current_epoch=self.current_epoch, batch_size = self.batch_size)
                if self.channel is not None:
                    ch = self.channel
                else:
                    ch = 0
                if self.sr_type is not None:
                    plot_super_resolution(self.model, self.to_predict, self.to_predict_hr, ch, self.fig_sr, self.axs_sr, self.nchs, loss, self.current_epoch, self.scales, self.scales_hr, mode = self.mode, batch_size=self.batch_size)
                #fig_latentspace = plot_latent_space(self.model, self.train_data, self.train_labels)
                #fig_pca_ls = plot_pca_latent_space(self.model, self.train_data, self.train_labels)
                self.fig.savefig('training_figures/reconstruction_{}epoch.png'.format(self.current_epoch), dpi=self.fig.dpi)
                self.fig_sr.savefig('training_figures/super_resolution_channel0_{}epoch.png'.format(self.current_epoch), dpi=self.fig_sr.dpi)
                #fig_latentspace.savefig('training_figures/latent_space_labels_{}epoch.png'.format(self.current_epoch), dpi=self.fig_latentspace.dpi)
                #fig_pca_ls.savefig('training_figures/pca_latent_space_labels_{}epoch.png'.format(self.current_epoch), dpi=self.fig_pca_ls.dpi)
                #plt.show()
                if "w" in self.mode: 
                    plot_wavelet_reconstruction(self.to_predict_signal, self.to_predict, self.scales, self.fig_recwav, self.axs_recwav,  fs = 50)
                    plot_wavelets_rt(self.model, self.to_predict, self.to_predict_hr, self.scales, self.scales_hr, self.fig_wv, self.axs_wv, self.fig_ecgw, self.axs_ecgw, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)
                    if self.sr_type is not None:
                        plot_wavelet_reconstruction(self.to_predict_hr_signal, self.to_predict_hr, self.scales_hr, self.fig_srwav, self.axs_srwav,  fs = 500)
                        plot_wavelets_sr_rt(self.model, self.to_predict, self.to_predict_hr, self.fig_wvsr, self.axs_wvsr, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)        
      
                    #dont save 
        print("Epoch: {}, Loss: {}".format(self.current_epoch, loss), end=' \r')
    
    def configure_optimizers(self):
       
        optimizer_rec = torch.optim.Adam(self.model.parameters(),
                               lr=self.params['LR']
                               )
        optims = [optimizer_rec]
        if self.sr_type is not None:
            optimizer_sr = torch.optim.Adam(self.model.parameters(),
                                   lr=self.params['LR_sr']
                                   )
            optims.append(optimizer_sr)
        if self.lr_scheduler:
            scheduler_rec = LinearLR(optimizer_rec, start_factor=1.0, end_factor=0.1, total_iters=30, verbose = True)
            schedulers = [scheduler_rec]
            if self.sr_type is not None:
                scheduler_sr = LinearLR(optimizer_sr, start_factor=1.0, end_factor=0.1, total_iters=30, verbose = True)
                schedulers.append(scheduler_sr)
            return optims, schedulers
        else:
            return optims 

class MMVAESR_Experiment3d(pl.LightningModule):

    def __init__(self,
                 vae_model,
                 params, 
                 mode,
                 to_predict,
                 label,
                 nchs,
                 batch_size,
                 channel = None,
                 to_predict_hr = None,
                 log_epochs = 10,
                 epochs_save_preliminar = 100,
                 loss_name = "mse",
                 lr_change = None,
                 lr_scheduler = False,
                 in_colab = False,
                 sr_type = "upsample",
                 loss_type = "lr" 
                ):
        
        super().__init__()

        self.params = params
        self.curr_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = vae_model.to(self.curr_device)
        self.mode = mode
        self.hold_graph = False
        self.log_epochs = log_epochs
        self.epochs_save_preliminar = epochs_save_preliminar
        self.to_predict = to_predict
        self.to_predict_hr = to_predict_hr
        self.label = label
        self.nchs = nchs
        self.batch_size = batch_size 
        self.loss_name = loss_name
        self.channel = channel 
        self.lr_scheduler = lr_scheduler
        self.in_colab = in_colab 
        self.automatic_optimization = False
        
        self.supported_sr_types = ["upsample", "convt", "none"]
        if sr_type == "none":
            sr_type = None
        elif sr_type not in self.supported_sr_types:
            sr_type = None
        self.sr_type = sr_type

        self.supporter_loss_types = ["lr", "lr+hr"]
        if loss_type not in self.supported_loss_types:
            loss_type = "lr"
        self.loss_type = loss_type 
        
        
        self.fig_sr, self.axs_sr = plt.subplots(3, figsize=(10, 8))
        
        
        if "w" in self.mode: 
            if self.to_predict_hr is None:
                raise Exception("to_predict_hr must be not None when ECG AE reconstruction and super resolution modality is wavelets based.")
            else:
                self.fig_wv, self.axs_wv = plt.subplots(self.nchs, 3, figsize=(10, 8))
                self.fig_wvsr, self.axs_wvsr = plt.subplots(1, 4, figsize=(10, 8))
                self.fig_ecgw, self.axs_ecgw = plt.subplots(self.nchs, 1, figsize=(10, 8))
                
        if self.nchs > 1:
            print(">1", self.nchs)
            self.fig, self.axs = plt.subplots(self.nchs, figsize=(10, 8))
        else:
            print("ELSE", self.nchs)
            self.fig, self.axs = plt.subplots(1, figsize=(10, 8))
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
    
    def forward(self, x, **kwargs):
        
        return self.model(x, **kwargs)

    """
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        #print(batch.shape)
        results = self.forward(batch)
        train_loss = self.model.loss_function(*results,
                                                  kld_weight = self.params['kld_weight'],
                                                  optimizer_idx=optimizer_idx,
                                                  batch_idx = batch_idx,
                                                 )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']
    """

    def training_step(self, batch, batch_idx):
        
        # Access the optimizer and scheduler
        if self.sr_type is not None:
            optimizer_rec_s, optimizer_rec_w , optimizer_sr_w = self.optimizers()
        else:
            optimizer_rec_s, optimizer_rec_w = self.optimizers()

        
        if self.lr_scheduler:
            if self.sr_type is not None:
                scheduler_rec_s, scheduler_rec_w, scheduler_sr = self.lr_schedulers()
            else:
                scheduler_rec_s, scheduler_rec_w = self.lr_schedulers()
        else:
            scheduler_rec_s = None
            scheduler_rec_w = None
            scheduler_sr = None

        # Compute loss
        
        results = self.forward(batch)
        loss = self.model.loss_function(*results, kld_weight = self.params['kld_weight'])
        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)

        # Perform reconstruction signal manual optimization
        optimizer_rec_s.zero_grad()
        loss_rec = loss["Reconstruction Loss Signal"]
        loss_rec.backward()
        optimizer_rec_s.step()

        # Perform reconstruction signal manual optimization
        optimizer_rec_w.zero_grad()
        loss_rec = loss["Reconstruction Loss Wavelets"]
        loss_rec.backward()
        optimizer_rec_w.step()

        if self.sr_type is not None:
            # Perform super resolution manual optimization
            if "hr" in self.loss_type:
                optimizer_sr_w.zero_grad()
                loss_sr = loss["Super Resolution Error Wavelets"]
                loss_sr.backward()
                optimizer_sr_w.step()

        # Optionally, update the learning rate using the scheduler
        if scheduler_rec_s is not None:
            scheduler_rec_s.step()
        if scheduler_rec_w is not None:
            scheduler_rec_w.step()
        if scheduler_sr is not None:
            scheduler_sr.step()
        
        return loss["loss"]
     
        
    def on_train_epoch_end(self):
        
        #print(self.curr_epoch)
        loss = self.logger.log_end_epoch() 
        if not self.in_colab:
            if self.current_epoch % self.log_epochs == 0 or self.current_epoch % self.epochs_save_preliminar == 0:
                clear_output()
                plot_prediction(self.model, self.to_predict, self.label, self.fig, self.axs, self.nchs, None,channel = self.channel, mode = self.mode, current_epoch=self.current_epoch, batch_size = self.batch_size)
                if self.channel is not None:
                    ch = self.channel
                else:
                    ch = 0
                if self.sr_type is not None:
                    plot_super_resolution(self.model, self.to_predict, self.to_predict_hr, ch, self.fig_sr, self.axs_sr, self.nchs, loss, self.current_epoch, None, None, mode = self.mode, batch_size=self.batch_size)
                if "w" in self.mode:
                    plot_wavelets_rt(self.model, self.to_predict, self.to_predict_hr, None, None, self.fig_wv, self.axs_wv, self.fig_ecgw, self.axs_ecgw, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)        
                    plot_wavelets_sr_rt(self.model, self.to_predict, self.to_predict_hr, self.fig_wvsr, self.axs_wvsr, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)        
                #fig_latentspace = plot_latent_space(self.model, self.train_data, self.train_labels)
                #fig_pca_ls = plot_pca_latent_space(self.model, self.train_data, self.train_labels)
                plt.show()

            if self.current_epoch % self.epochs_save_preliminar == 0:
                clear_output() 
                plot_prediction(self.model, self.to_predict, self.label, self.fig, self.axs, self.nchs, None, channel = self.channel, mode = self.mode, current_epoch=self.current_epoch)
                if self.channel is not None:
                    ch = self.channel
                else:
                    ch = 0
                if self.sr_type is not None:
                    plot_super_resolution(self.model, self.to_predict, self.to_predict_hr, ch, self.fig_sr, self.axs_sr, self.nchs, loss, self.current_epoch, None,  None, mode = self.mode, batch_size=self.batch_size)
                #fig_latentspace = plot_latent_space(self.model, self.train_data, self.train_labels)
                #fig_pca_ls = plot_pca_latent_space(self.model, self.train_data, self.train_labels)
                self.fig.savefig('training_figures/reconstruction_{}epoch.png'.format(self.current_epoch), dpi=self.fig.dpi)
                self.fig_sr.savefig('training_figures/super_resolution_channel0_{}epoch.png'.format(self.current_epoch), dpi=self.fig_sr.dpi)
                #fig_latentspace.savefig('training_figures/latent_space_labels_{}epoch.png'.format(self.current_epoch), dpi=self.fig_latentspace.dpi)
                #fig_pca_ls.savefig('training_figures/pca_latent_space_labels_{}epoch.png'.format(self.current_epoch), dpi=self.fig_pca_ls.dpi)
                #plt.show()
                if "w" in self.mode: 
                    plot_wavelets_rt(self.model, self.to_predict, self.to_predict_hr, self.scales, self.scales_hr, self.fig_wv, self.axs_wv, self.fig_ecgw, self.axs_ecgw, nchs = self.nchs, batch_size=self.batch_size, epoch = self.current_epoch)
                    #dont save 
        print("Epoch: {}, Loss: {}".format(self.current_epoch, loss), end=' \r')
    
    def configure_optimizers(self):
       
        optimizer_rec_w = torch.optim.Adam(self.model.parameters(),
                               lr=self.params['LR']
                               )
        optimizer_rec_s =  torch.optim.Adam(self.model.parameters(),
                               lr=self.params['LR']
                               )
        optims = [optimizer_rec_s, optimizer_rec_w]
        if self.sr_type is not None:
            optimizer_sr_w = torch.optim.Adam(self.model.parameters(),
                                   lr=self.params['LR']
                                   )
            optims.append(optimizer_sr_w)
        if self.lr_scheduler:
            scheduler_rec_s = LinearLR(optimizer_rec_s, start_factor=1.0, end_factor=0.5, total_iters=30)
            scheduler_rec_w = LinearLR(optimizer_rec_w, start_factor=1.0, end_factor=0.5, total_iters=30)
            schedulers = [scheduler_rec_s, scheduler_rec_w]
            if self.sr_type is not None:
                scheduler_sr_w = LinearLR(optimizer_sr_w, start_factor=1.0, end_factor=0.5, total_iters=30)
                schedulers.append(scheduler_sr_w)
            return optims, schedulers
        else:
            return optims 

def torch_removebyindex(torch_array, idx):
    torch_array = torch.cat([torch_array[:idx], torch_array[idx+1:]])
    return torch_array

import collections

from pytorch_lightning.loggers import Logger 
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

class HistoryLogger(Logger):
    def __init__(self):
        super().__init__()

        self.history = collections.defaultdict(list) # copy not necessary here
        self.log = collections.defaultdict(list) 
        self.loss = collections.defaultdict(list) 
        self.epoch = 1
        self.metric_names = []
        # The defaultdict in contrast will simply create any items that you try to access

    @property
    def name(self):
        return "Logger_custom_plot"

    @property
    def version(self):
        return "1.0"

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        
        for metric_name, metric_value in metrics.items():
            if metric_name != 'epoch':
                if metric_name not in self.metric_names:
                    self.metric_names.append(metric_name)
                self.history["temp_"+metric_name].append(metric_value)
            
    def log_end_epoch(self): 

        for metric_name in self.metric_names:
            epoch_metric_values = self.history["temp_"+metric_name]
            mean_metric = np.mean(epoch_metric_values)
            self.log[metric_name].append(mean_metric)
            self.loss[self.epoch] = epoch_metric_values
            self.epoch += 1
        return self.log["loss"][-1]
    
    def log_hyperparams(self, params):
        pass

def resample(to_resample, size = 2048):
    
    n = to_resample.shape[0]
    nchs = 12
    resampled_data = np.zeros((n, nchs, size))
    for i, data in enumerate(to_resample):
        for ch in range(nchs):
            resampled_data[i, ch, :] = scipy.signal.resample(x=data[ch, :], num=size) 
    
    return resampled_data

def prepare_training_sr_gif(epochs, epochs_save_preliminar):
    epochs_toplot = np.arange(epochs_save_preliminar, epochs+1, epochs_save_preliminar)
    directory = "training_figures/" 
    images = []
    for epoch in epochs_toplot:
        filename = directory+"super_resolution_channel0_{}epoch.png".format(epoch)
        images.append(imageio.imread(filename))

    imageio.mimsave(directory+"sr_reconstruction_0.gif", images)

def prepare_training_gif(epochs, epochs_save_preliminar):
    epochs_toplot = np.arange(epochs_save_preliminar, epochs+1, epochs_save_preliminar)
    directory = "training_figures/" 
    images = []
    for epoch in epochs_toplot:
        filename = directory+"reconstruction_{}epoch.png".format(epoch)
        images.append(imageio.imread(filename))

    imageio.mimsave(directory+"reconstruction.gif", images)

def plot_channels_data(id, datas):
    
    data = datas[id]
    nchs = 12
    n = data.shape[-1]
    t = np.arange(0, n, 1)
    sig_or = torch.squeeze(data).cpu().detach().numpy()
    fig, axs = plt.subplots(nchs, figsize=(20, 40))

    for j in range(nchs):
        axs[j].plot(t, sig_or[j, :], "g")
        axs[j].title.set_text("Channel {}".format(j+1))

def check_super_resolution_quality(not_resampled_data, super_resolution_data):
    
    mse_errors = []
    for i, data in enumerate(not_resampled_data):
        print(i+1, "/", len(not_resampled_data), end="\r")
        sr_data = super_resolution_data[i]
        error = F.mse_loss(data, sr_data)
        mse_errors.append(error.item())
        del error
    mean_mse = np.mean(mse_errors)
    return mean_mse, mse_errors

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal.T for signal, meta in data])
    return data

def sliding_window(signal, size=250, stride=250):
    windows = []
    sig_len = signal.shape[-1]
    win_num = math.ceil((sig_len - size) / stride) + 1
    for i in range(win_num):
        offset = i * stride
        windows.append((signal[:, offset : offset + size]))
    return windows


def split_windows(data, labels, width = 250, stride = 250):
    windows_data = []
    windows_labels = []
    for i, signal in enumerate(data):
        label = labels[i]
        signal = torch.squeeze(signal)
        windows = sliding_window(signal, width, stride = stride)
        for window in windows:
            windows_data.append(window)    
            windows_labels.append(label)
            
    temp = torch.zeros(len(windows_data), 1, 12, width)
    temp_labels = torch.zeros(len(windows_data))
    for i, data in enumerate(windows_data):
        label = windows_labels[i]
        temp[i, :, :, :] = torch.unsqueeze(data, dim = 0)
        temp_labels[i] = label
        
    windows_data = temp
    windows_labels = temp_labels
    return windows_data, windows_labels 


def train_1d_model(train_dataloader, width, to_predict, img, label, batch_size, kernel_sizes = None, strides = None, outchannels = None, lr_scheduler = False, last_tanh = True, model_type = "ae", epochs_save_preliminar = 10, do = 0.2, activation_name = "sigmoid", epochs = 100, lr = 0.0001, mode = "s", channel = None, lr_change = {100: 0.00005}, loss_name = "mse", in_colab = False, early_stopping = False, sr_type = None, kernel_sizes_sr = None, of_sr = None, str_sr = None, loss_type = "lr", to_predict_hr = None, denoising = True, kld_weight= 1.0, latent_dim = None):
    
    if kernel_sizes is None or strides is None or outchannels is None:
        raise Exception("One of 'kernel_sizes', 'strides' or 'outchannels' parameter is None, please set those parameters before run.")
            
    nchs = 1 if channel is not None else 12
        
    args = {"batch_num": 1, "log_interval": 10}
    to_choice = list(np.arange(1, nchs))
  
    num_layers = len(kernel_sizes)
    toprod =  [item for items in kernel_sizes for item in items]#item*2
    div = np.sum([elem for elem in toprod if elem >= 1])
    
    if latent_dim  is None:
        latent_dim = int((width - div + num_layers*2) * outchannels[-1][-1])
    print(outchannels[-1][-1])

    size = width * 10
    #print(len(kernel_sizes), len(out_channels))
    print("Latent dim: ", latent_dim)
    
    model = VAE1d_SR_multimodal(to_predict, img, batch_size = batch_size, num_layers=num_layers, type = model_type, in_channels=nchs, out_filters = outchannels, last_tanh = last_tanh, strides = strides, do = do, size = size, latent_dim = latent_dim, width = width, device = device, kernel_sizes=kernel_sizes, mode=mode, activation_name = activation_name, loss_name = loss_name, sr_type = sr_type, kernel_sizes_sr = kernel_sizes_sr, str_sr = str_sr, of_sr = of_sr, loss_type = loss_type, denoising = denoising)
    model_params = {"LR": lr, "kld_weight": kld_weight}#<1 to prioritize reconstruction
    callbacks = []
    if early_stopping:
        early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.000001, patience=10, verbose=False, mode='min')
        callbacks.append(early_stop_callback)
    logger = HistoryLogger()
    
    log_epochs = 1#int(epochs/10)

    model_run = MMVAESR_Experiment(model, model_params, mode, to_predict, label, nchs, channel = channel, batch_size = batch_size, log_epochs=log_epochs, epochs_save_preliminar = epochs_save_preliminar, lr_change = lr_change, loss_name = loss_name, lr_scheduler=lr_scheduler, in_colab = in_colab, loss_type = loss_type, to_predict_hr = to_predict_hr, sr_type = sr_type)
    if lr_scheduler:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    log_every_n_steps = 1
    trainer1d = pl.Trainer(accelerator=accelerator, max_epochs=epochs, callbacks=callbacks, logger= logger, log_every_n_steps = log_every_n_steps)#, accumulate_grad_batches=batch_size)
    trainer1d.fit(model_run, train_dataloaders=train_dataloader)
    
    model = model.to(device)
    
    if channel is None:
        torch.save(model, "models/model_{}.pt".format(label))
    else:
        torch.save(model, "models/model_{}_channel{}.pt".format(label, channel))
        
    return model, trainer1d, model_run

def train_2d_model(train_dataloader, to_predict, to_predict_hr, scales, scales_hr, img, label, batch_size, kernel_sizes = None, strides = None, outchannels = None, model_type = "ae", epochs_save_preliminar = 10, do = 0.2, activation_name = "sigmoid", size = None, epochs = 100, lr = 0.001, lr_sr = 0.0001, mode = "s", channel = None, lr_change = {100: 0.00005}, loss_name = "mse", lr_scheduler = False, in_colab = False, sr_type = "upsample", early_stopping = False, loss_type = "lr", ks_sr = None, str_sr = None, of_sr = None,  kernel_sizes_s = None, out_channels_s=None, strides_s = None, kernel_sizes_sr_s=None, str_sr_s = None, of_sr_s = None, to_predict_s = None, to_predict_s_hr=None, kld_weight = 1.0):
    
    if kernel_sizes is None or strides is None or outchannels is None:
        raise Exception("One of 'kernel_sizes', 'strides' or 'outchannels' parameter is None, please set those parameters before run.")
    
    if mode == "sw":
        if kernel_sizes_s is None or strides_s is None or out_channels_s is None:
            raise Exception("One of 'kernel_sizes_s', 'strides_s' or 'out_channels_s' parameter is None, please set those parameters before run.")
        if to_predict_s is None or to_predict_s_hr is None:
            raise Exception("One of 'to_predict_s' or 'to_predict_s_hr' parameter is None, please set those parameters before run.")
            
    nchs = 1 if channel is not None else 12
     
    #acc_batch_size = batch_size

    args = {"batch_num": 1, "log_interval": 10}
    to_choice = list(np.arange(1, nchs))
  
    num_layers = len(kernel_sizes)
    width = to_predict.shape[-1]
    height = to_predict.shape[-2]
 
   
    #size = width * 10
    if size is None:
        size = [125, width*10]

    model = VAE2d_SR_multimodal(to_predict, to_predict_hr = to_predict_hr, img = img, batch_size = batch_size, num_layers=num_layers, type = model_type, sr_type = sr_type, in_channels=nchs, out_filters = outchannels, strides = strides, do = do, size = size, width = width, device = device, kernel_sizes=kernel_sizes, mode=mode, activation_name = activation_name, loss_name = loss_name, loss_type = loss_type, ks_sr = ks_sr, str_sr = str_sr, of_sr = of_sr,  kernel_sizes_s = kernel_sizes_s, out_filters_s=out_channels_s, strides_s = strides_s, ks_s_sr=kernel_sizes_sr_s, str_s_sr = str_sr_s, of_s_sr = of_sr_s, to_predict_s = to_predict_s, to_predict_s_hr=to_predict_s_hr)
    
    model_params = {"LR": lr, "LR_sr": lr_sr, "kld_weight": kld_weight}#<1 to prioritize reconstruction
    callbacks = [] 
    if early_stopping:
        early_stop_callback = EarlyStopping(monitor='loss', min_delta=0.000001, patience=10, verbose=False, mode='min')
        callbacks.append(early_stopping)
    logger = HistoryLogger()
    
    log_epochs = 1#int(epochs/10)
    
    model_run = MMVAESR_Experiment2d(model, model_params, mode, to_predict, label, nchs, scales = scales, scales_hr = scales_hr, to_predict_hr = to_predict_hr, batch_size = batch_size, log_epochs=log_epochs, sr_type = sr_type, epochs_save_preliminar = epochs_save_preliminar, lr_change = lr_change, loss_name = loss_name, lr_scheduler = lr_scheduler, in_colab = in_colab, channel = channel, loss_type = loss_type, to_predict_signal = to_predict_s, to_predict_hr_signal = to_predict_s_hr)
    
    if lr_scheduler:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    log_every_n_steps = 1
    trainer2d = pl.Trainer(accelerator=accelerator, max_epochs=epochs, callbacks=callbacks, logger= logger, log_every_n_steps = log_every_n_steps)#, accumulate_grad_batches=acc_batch_size)
    trainer2d.fit(model_run, train_dataloaders=train_dataloader)
    
    model = model.to(device)
    
    if channel is None:
        torch.save(model, "models/model2d_{}.pt".format(label))
    else:
        torch.save(model, "models/model2d_{}_channel{}.pt".format(label, channel))
        
    return model, trainer2d, model_run

def plot_super_resolution_final(data_hr, data_sr, channel = None):

    n = data_hr.shape[-1]
    t = np.arange(0, n, 1)
    
    if channel is None:
        fig, axs = plt.subplots(12, 1, figsize = (20, 20))
        nchs = 12
    else:
        fig, axs = plt.subplots(1, 1, figsize = (20, 20))
        nchs = 1

    sig_or = data_hr
    sig_recons = data_sr
    
    loss = F.mse_loss(sig_or, sig_recons).item()#device
    fig.suptitle("MSE Loss: {}".format(loss))
    
    
    #print(sig_or.shape, sig_recons.shape)
    sig_recons = sig_recons.cpu().detach().numpy()
    sig_or = sig_or.cpu().detach().numpy()

    if channel is None:
        
        fig.subplots_adjust(wspace=0, hspace=1)
        #fig.tight_layout(pad=0.000)
        for j in range(nchs):
            
            y_min = min([min(sig_or[0, j, :]), min(sig_recons[0, j, :])])
            y_max = max([max(sig_or[0, j, :]), max(sig_recons[0, j, :])])
            
            for line in axs[j].get_lines(): # ax.lines:
                line.remove()
            if j == 0:
                axs[j].plot(t, sig_or[0, j, :], "g", label = "high resolution 500H")
                axs[j].plot(t, sig_recons[0, j, :], "b", label =  "reconstructed super resolution 500Hz")
            else:
                axs[j].plot(t, sig_or[0, j, :], "g")
                axs[j].plot(t, sig_recons[0, j, :], "b")
            axs[j].title.set_text("Channel {}".format(j+1))
            axs[j].set_ylim([y_min, y_max])
    else:
        
        y_min = min([min(sig_or[0, 0, :]), min(sig_recons[0, 0, :])])
        y_max = max([max(sig_or[0, 0, :]), max(sig_recons[0, 0, :])])
        
        for line in axs.get_lines(): # ax.lines:
            line.remove()
        axs.plot(t, sig_or[0, 0, :], "g", label = "high resolution 500H")
        axs.plot(t, sig_recons[0, 0, :], "b", label =  "reconstructed super resolution 500Hz")
        axs.title.set_text("Channel {}".format(channel))
        axs.set_ylim([y_min, y_max])
    
    fig.legend()

def plot_diff_superresolution(hr_data, sr_data):
    
    loss = F.mse_loss(hr_data.cpu(), sr_data.cpu())
    hr_data = torch.squeeze(hr_data, dim = 0).cpu().detach().numpy() 
    sr_data = torch.squeeze(sr_data, dim = 0).cpu().detach().numpy() 

    if hr_data.shape != sr_data.shape:
        raise Exception("hr data and sr data must have the same shape but different shapes are given. Shapes {} for hr_data and {} for sr_data were given".format(hr_data.shape, sr_data.shape))
    else:
        width = hr_data.shape[1]
        nchs = hr_data.shape[0]
        
    diff = np.zeros((nchs, width))
    for ch in range(nchs):
        hr_data_ch = hr_data[ch]
        sr_data_ch = sr_data[ch]
        diff[ch] = hr_data_ch - sr_data_ch
    
    fig, axs = plt.subplots(nchs, 1, figsize= (20, 20))
    fig.suptitle("Difference between HR 500 Hz signal and reconstructed SR signal at 500 Hz, MSE : {}".format(loss))
    for ch in range(nchs):
        ymin = min([min(hr_data), min(sr_data)])
        ymax = max([max(hr_data), max(sr_data)])
        axs[ch].plot(diff[ch], "r")
        axs[ch].set_title("Channel {}".format(ch))
        axs[ch].set_ylim([ymin, ymax])
    fig.tight_layout()

def plot_loss(trainer, model_type, epochs):
        
    if model_type == "vae":
        losses = trainer.logger.log["loss"]
        kld = trainer.logger.log["KLD"]
    mse_loss = trainer.logger.log["Reconstruction Loss Signal"]#add super resolution error

    epochs_arr = np.arange(epochs)

    fig, ax1 = plt.subplots(figsize = (20, 5))
    ax1.plot(epochs_arr, mse_loss, "r", label = "MSE", linewidth = 2)
    ax1.set_ylabel("MSE loss", color="r", size = 28) 
    if model_type == "vae":
        ax2 = ax1.twinx()
    
    plt.title("Train Losses", size = 36)
    plt.xlabel("Epochs", size = 28)

    ax1.set_xticks(np.arange(0, epochs, int(epochs/10)))
    if model_type == "vae":
        ax2.plot(epochs_arr, losses, "k", label = "total loss", linewidth = 2)
        ax2.tick_params(axis='y', labelcolor="r")

        color = 'black'

        ax2.set_ylabel('kld - total loss', color=color, size = 28) 
        ax2.plot(epochs_arr, kld, color = "blue", label = "kld", linewidth = 2)
        ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    fig.legend(loc="upper right")
    

class SignalDataset(Dataset):
    def __init__(self, signals, labels, sublabels, signals_hr = None):
        self.labels = labels
        self.signals = signals 
        self.signals_hr = signals_hr
        self.sublabels = sublabels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        sublabel = self.sublabels[idx]
        if self.signals_hr is not None:
            signal_hr = self.signals_hr[idx]
            return signal, signal_hr, label, sublabel
        else:
            return signal, label, sublabel

class WaveletDataset(Dataset):
    def __init__(self, wavelets, wavelets_hr, scales, scales_hr, labels, sublabels):
        self.labels = labels
        self.wavelets = wavelets
        self.wavelets_hr = wavelets_hr
        self.scales = scales
        self.scales_hr = scales_hr
        self.sublabels = sublabels
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wavelet_lr = self.wavelets[idx].float().to(self.device)
        wavelet_hr = self.wavelets_hr[idx].float().to(self.device)
        scale = self.scales[idx].float().to(self.device)
        scale_hr = self.scales_hr[idx].float().to(self.device)
        label = self.labels[idx]
        sublabel = self.sublabels[idx]
        return wavelet_lr, wavelet_hr, scale, scale_hr, label, sublabel

class EfficientWaveletDataset(Dataset):
    def __init__(self, signals, signals_hr, labels, sublabels, nchs = 12, mode = "w"):
        self.labels = labels
        self.signals = signals
        self.signals_hr = signals_hr
        self.nchs = nchs 
        self.device = device 
        self.sublabels = sublabels
        
        
        modes = ["w", "sw", "ws"]
        self.mode = mode
        if self.mode == "ws":
            self.mode = "sw"
            
        if self.mode not in modes:
            self.mode = "w"
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        #gc.collect() #slow up the process 
        torch.cuda.empty_cache()
        
        signal_lr = self.signals[idx]
        signal_hr = self.signals_hr[idx]
        #print("lr")
        wavelet_lr, scale_lr = compute_single_wavelet(signal_lr, fs = 50, nchs = self.nchs)
        #print("hr")
        wavelet_hr, scale_hr  = compute_single_wavelet(signal_hr, fs = 500, nchs = self.nchs)
        
        label = self.labels[idx]
        sub_label = self.sublabels[idx]
        
        if self.mode == "w":
            return wavelet_lr.float().to(self.device), wavelet_hr.float().to(self.device), scale_lr, scale_hr, label, sub_label
        else:
            return signal_lr.to(self.device), signal_hr.to(self.device), wavelet_lr.float().to(self.device), wavelet_hr.float().to(self.device), scale_lr, scale_hr, label, sub_label
        
def activation_layer(name):
     
    name = name.lower()
    activations = {
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "leakyrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "gelu": nn.GELU()    
    }
    if name in activations.keys():
        return activations[name]
    else:
        print("Activation function {} not supported. Using Sigmoid".format(name))
        return nn.Sigmoid()

def compute_single_wavelet(signal, fs = 50, nchs = 12):
    
    wave = []
    if fs == 50:
        dj = 0.5
        height = 15
    else:
        dj = 0.1 
        height = 104
    dt = 1/fs

    width = signal.shape[-1]
    wave = np.zeros((nchs, height, width), dtype = np.float64) #HERE THE PROBLEM WHEN NWORKERS != None
 
    for ch in range(nchs):
        
        #scale = np.arange(1, 15, dj)
        #print(signal[0, ch].shape)
        if nchs > 1:
            if signal.ndim == 3:
                signal = signal[0]
            s_ch = signal[ch].detach().cpu().numpy()
        else:
            if signal.ndim == 2:
                signal = signal[0]
            s_ch = signal[:].detach().cpu().numpy()
        wav, scale, _, _, _, _ =  pycwt.cwt(s_ch, dt, dj = dj, wavelet = "morlet")
        wave[ch, :, :] = wav
        #scales = scale
    
    wave = torch.from_numpy(wave).float()#.to(device)
    
    return wave, scale

def get_dataloader(dict_windows_lr, dict_windows_hr, sub_labels, channel=None, label="NORM", input = "s", loss_type = "lr+hr", quicktest = False, n_workers = None, batch_size = 1, dict_wav_lr = None, dict_wav_hr = None, denoising = True, precomputed = False):


    data = dict_windows_lr[label]
    sub_labels = sub_labels[label]
    
    supported_input = ["s", "w", "sw", "ws"]
        
    if "hr" in loss_type:
        hr = True
    else:
        hr = False
    
    
    if hr:
        if dict_windows_hr is None:
            raise Exception("Loss type 'hr' mode requires both Low Resolution and High resolution data in input, but HR data is missing.")
        if dict_windows_lr is None:
            raise Exception("Loss type 'hr' mode requires both Low Resolution and High resolution data in input, but LR data is missing.")

    if input in supported_input:
        if "s" in input:
            nchs = data.shape[-2]
            if hr:
                data_hr = dict_windows_hr[label]
                width_hr = data_hr.shape[-1]
            else:
                if denoising:
                    data_lr_clean =  dict_windows_hr[label]
                    if data_lr_clean.shape[-1] != data.shape[-1]:
                        raise Exception("Shape missmatch during Denoising CAE: {} != {}".format(data_lr_clean.shape, data.shape))
            
        elif input == "w":
            data_hr = dict_windows_hr[label]
            width_hr = data_hr.shape[-1]
            nchs = data.shape[-3]
            height = data.shape[-2]
            if dict_wav_lr is not None:
                data_wav = dict_wav_lr[label]
            else: 
                data_wav = None
            if dict_wav_hr is not None:
                data_wav_hr = dict_wav_hr[label]
            else:
                data_wav_hr = None
    width = data.shape[-1]

    if channel is not None:
        if "s" in input:
            data = data[:, :, channel, :]
            if hr:
                data_hr = data_hr[:, :, channel, :]
            else:
                if denoising: 
                    data_lr_clean = data_lr_clean[:, :, channel, :]
        elif input == "w":
            data = data[:, channel, :, :]
            data_hr = data_hr[:, channel, :, :]
            
        if data_wav is not None:
            data_wav = data_wav[:, channel, :, :]
        if data_wav_hr is not None:
            data_wav_hr = data_wav_hr[:, channel, :, :]
            
    else:
        if "s" in input:
            data = torch.squeeze(data, dim=1)
            if hr:
                data_hr = torch.squeeze(data_hr, dim=1)
        
    if quicktest:
        data = data[:256]#for quick testing only
        if hr:
            data_hr = data_hr[:256]
        if input == "w":
            data_hr = data_hr[:256]
            
    data = data.to(device)
    if input == "w" or hr:
        data_hr = data_hr.to(device)
    label_num = map_superclass_rev[label]

    n = data.shape[0]
    y_label = [label_num]*n
    labels = torch.FloatTensor(y_label).to(device)

    if input == "s":
        if hr:
            dataset = SignalDataset(data, labels, sub_labels, signals_hr = data_hr)
        else:
            if denoising:
                dataset = SignalDataset(data, labels, sub_labels, signals_hr = data_lr_clean)
            else:
                dataset = SignalDataset(data, labels, sub_labels, signals_hr = None)
    
    elif "w" in input:
        if channel is None:
            nchs = 12
        else:
            nchs = 1
        
        if not precomputed:
            dataset = EfficientWaveletDataset(data, data_hr, labels, sub_labels, nchs = nchs, mode = input)
        else:
            if "s" in input:
                dataset = WaveletDataset(data_wav, data_wav_hr, labels, sub_labels, nchs = nchs, data = data, data_hr = data_hr)
            else:
                dataset = WaveletDataset(data_wav, data_wav_hr, labels, sub_labels, nchs = nchs)

    if n_workers is None:
        dataloader = DataLoader(dataset, batch_size = batch_size)
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers = n_workers)
    to_predict = next(iter(dataloader))
    if input == "w":
        to_predict_lr, to_predict_hr, scales, scales_hr, label, sub_label  = to_predict
        return dataloader, to_predict_lr, to_predict_hr, scales, scales_hr, label, sub_label
    elif input == "s":
        if hr:
            to_predict_lr, to_predict_hr, label, sub_label = to_predict
            return dataloader, to_predict_lr, to_predict_hr, label, sub_label
        else:
            if denoising:
                to_predict_lr, to_predict_lr_clean, label, sub_label = to_predict
                return dataloader, to_predict_lr, to_predict_lr_clean, label, sub_label
            else:
                to_predict_lr, label, sub_label = to_predict
                return dataloader, to_predict_lr, None, label, sub_label
    else:
        to_predict_s_lr, to_predict_s_hr, to_predict_w_lr, to_predict_w_hr, scales, scales_hr, label, sub_label  = to_predict
        return dataloader, to_predict_s_lr, to_predict_s_hr, to_predict_w_lr, to_predict_w_hr, scales, scales_hr, label, sub_label
        
def plot_wavelets(wavelets, datas, nchs = 12, fs_lr = 50, fs_hr = 500, fs_sr = 500, ch = None):
    

    titles = ["Low Resolution GMW wavelets", "High Resolution GMW wavelets", "Super Resolution GMW wavelets"]
    titles_data = ["High Resolution signal at {}Hz".format(fs_hr), "Super Resoluted {}Hz signal from low resolution {}Hz".format(fs_sr, fs_lr)]
    datas = [data.cpu() for data in datas]
    
    lr_wave = wavelets[0]
    if ch is None:
        for ch in range(nchs):
            fig1, axs = plt.subplots(1, 3, figsize = (20, 5))
            fig1.suptitle("Wavelets Channel {}".format(ch))
            imgs = []
            vmaxs = []
            for j in range(3):
                if nchs != 1:
                    img = axs[j].imshow(np.abs(wavelets[j][ch]), aspect='auto', cmap='turbo')    
                else:
                    img = axs[j].imshow(np.abs(wavelets[j]), aspect='auto', cmap='turbo') 

                divider = make_axes_locatable(axs[j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig1.colorbar(img, cax=cax, orientation='vertical')

                # Assume colorbar was plotted last one plotted last
                cb = img.colorbar   
                vmaxs.append(cb.vmax)
                imgs.append(img)
                cb.remove()
                axs[j].set_title(titles[j])

            idx = np.argmax(vmaxs)
            divider = make_axes_locatable(axs[-1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig1.colorbar(imgs[idx], cax=cax, orientation='vertical')

            fig2 = plt.figure(figsize = (20, 5))
            fig2.suptitle("ECG Channel {}".format(ch))
            colors = ["g", "r--"]
            labels = ["high resolution", "super resoluted from low resolution"]
            widths = [4, 2]
            for j in range(2):
                if nchs != 1:
                    plt.plot(datas[j][ch, :], colors[j], label = labels[j], linewidth = widths[j])
                else:
                    print(datas[j].shape)
                    plt.plot(datas[j], colors[j], label = labels[j], linewidth = widths[j])
            plt.legend()
            print("Mean Squared Error: %.3g" % np.mean(np.abs(datas[1] - datas[0])**2))
        
    else:
        
        fig1, axs = plt.subplots(1, 3, figsize = (20, 5))
        fig1.suptitle("Wavelets Channel {}".format(ch))
        imgs = []
        vmaxs = []
        for j in range(3):
            wav = wavelets[j]
            if wav.ndim == 4:
                wav = torch.squeeze(wav, dim = 0)
            if wav.ndim == 3:
                wav = torch.squeeze(wav, dim = 0)
            wav = wav.detach().numpy()
            img = axs[j].imshow(np.abs(wav[ch]), aspect='auto', cmap='turbo')    
                
            divider = make_axes_locatable(axs[j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig1.colorbar(img, cax=cax, orientation='vertical')

            # Assume colorbar was plotted last one plotted last
            cb = img.colorbar   
            vmaxs.append(cb.vmax)
            imgs.append(img)
            cb.remove()
            axs[j].set_title(titles[j])

        idx = np.argmax(vmaxs)
        divider = make_axes_locatable(axs[-1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig1.colorbar(imgs[idx], cax=cax, orientation='vertical')

        fig2 = plt.figure(figsize = (20, 5))
        fig2.suptitle("ECG Channel {}".format(ch))
        colors = ["g", "r--"]
        labels = ["high resolution", "super resoluted from low resolution"]
        widths = [4, 2]
        for j in range(2):
            plt.plot(datas[j][ch, :], colors[j], label = labels[j], linewidth = widths[j])
        plt.legend()
        print("Mean Squared Super Resolution Error: %.3g" % torch.mean(torch.abs(datas[1] - datas[0])**2))
    return fig1, fig2
        
def get_cv2_img(to_predict, channel=None):
    #plt.rcParams['axes.facecolor']='black'
    plt.rcParams['axes.facecolor']="white"
    
    if to_predict.ndim == 4:
        to_predict = torch.squeeze(to_predict, dim = 0)
        
    n = to_predict.shape[-1]
    t = np.arange(0, n, 1)

    # redraw the canvas
    fig, ax = plt.subplots(figsize = (20, 5))
    if channel is None:
        plt.plot(t, to_predict[0, 0, :].cpu().detach().numpy().flatten(), "k", linewidth = 6)
    else:
        plt.plot(t, to_predict[0, :].cpu().detach().numpy().flatten(), "k", linewidth = 6)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    fig.patch.set_visible(False)
    #ax.set_facecolor('black')
    fig.show()
    return fig

def get_cv2_img(to_predict, channel=None):
    #plt.rcParams['axes.facecolor']='black'
    plt.rcParams['axes.facecolor']="white"
    
    if to_predict.ndim == 4:
        to_predict = torch.squeeze(to_predict, dim = 0)
        
    n = to_predict.shape[-1]
    t = np.arange(0, n, 1)

    # redraw the canvas
    fig, ax = plt.subplots(figsize = (20, 5))
    if channel is None:
        plt.plot(t, to_predict[0, 0, :].cpu().detach().numpy().flatten(), "k", linewidth = 6)
    else:
        plt.plot(t, to_predict[0, :].cpu().detach().numpy().flatten(), "k", linewidth = 6)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    fig.patch.set_visible(False)
    #ax.set_facecolor('black')
    fig.show()
    return fig

def get_loss_function(name):
    supported_loss = ["rpeakmse", "mse", "bce"]
    name = name.lower()
    if name not in supported_loss:
        name = "mse"
        
    if name == "mse":
        lossf = nn.MSELoss()
    elif name == "rpeakmse":
        lossf = PeakDetectionLoss(2, 1, 10)

    lossf.requires_grad_ = True
    return lossf

