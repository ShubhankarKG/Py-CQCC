from distutils.log import error
from math import ceil, log2
import sys
import numpy as np
from CQT_Toolbox.cqt import cqt
import librosa
import scipy

def Deltas(x, hlen):
    win = list(range(hlen, -hlen-1, -1))
    
    xx_1 = np.tile(x[:,0],(1,hlen)).reshape(hlen,-1).T
    xx_2 = np.tile(x[:,-1],(1,hlen)).reshape(hlen,-1).T
    
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    
    D = scipy.signal.lfilter(win, 1, xx)

    D = D[:,hlen*2:]
    D = D /(2*sum(np.arange(1,hlen+1))**2)

    return D

def cqcc(*args):
    nargin = len(args)

    if nargin < 2:
        error("Not enough arguments")
        sys.exit(1)
    
    x, fs = args[0], args[1]

    B = 96 if nargin < 3 else int(args[2])
    
    fmax = fs/2 if nargin < 4 else int(args[3])
    
    if nargin < 5:
        oct = ceil(log2(fmax/20))
        fmin = fmax / 2**oct
    else :
        fmin = int(args[4])
    
    d = 16 if nargin < 6 else int(args[5])

    cf = 19 if nargin < 7 else int(args[6])

    ZsdD = 'ZsdD' if nargin < 8 else args[7]

    gamma = 228.7 * (2 ** (1/B) - 2 ** (-1/B))

    # CQT Computing
    Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma)

    # Log Power Spectrum
    absCQT = np.abs(Xcq['c'])
    
    # TimeVec = np.arange(1, absCQT.shape[1]+1).reshape(1, -1)
    # TimeVec = TimeVec*Xcq['xlen'] / absCQT.shape[1] / fs

    # FreqVec = np.arange(0, absCQT.shape[0]).reshape(1, -1)
    # FreqVec = fmin * 2 ** (FreqVec / B)

    eps = 2.2204e-16
    LogP_absCQT = np.log(absCQT ** 2 + eps)

    # Uniform Resampling
    kl = B * log2(1+1/d)
    Ures_LogP_absCQT = librosa.resample(
        LogP_absCQT.T,
        fs,
        9_562
    ).T
    Ures_FreqVec = None

    # DCT
    CQcepstrum = scipy.fftpack.dct(Ures_LogP_absCQT, type=2, axis=1, norm='ortho')

    # Dynamic Coefficients
    if 'Z' in ZsdD:
        scoeff = 1
    else: 
        scoeff = 2
    CQcepstrum_temp = CQcepstrum[scoeff-1:cf+1,:]
    f_d = 3

    if ZsdD.replace('Z', '') == 'sdD':
        CQcc = np.concatenate([
            CQcepstrum_temp,
            Deltas(CQcepstrum_temp, f_d),
            Deltas(Deltas(CQcepstrum_temp, f_d), f_d)
        ], axis=0)
    elif ZsdD.replace('Z', '') == 'sd':
        CQcc = np.concatenate([
            CQcepstrum_temp,
            Deltas(CQcepstrum_temp, f_d)
        ], axis=0)
    elif ZsdD.replace('Z', '') == 'sD':
        CQcc = np.concatenate([
            CQcepstrum_temp,
            Deltas(Deltas(CQcepstrum_temp, f_d), f_d)
        ], axis=0)
    elif ZsdD.replace('Z', '') == 's':
        CQcc = CQcepstrum_temp
    elif ZsdD.replace('Z', '') == 'd':
        CQcc = Deltas(CQcepstrum_temp, f_d)
    elif ZsdD.replace('Z', '') == 'D':
        CQcc = Deltas(Deltas(CQcepstrum_temp, f_d), f_d)
    elif ZsdD.replace('Z', '') == 'dD':
        CQcc = np.concatenate([
            Deltas(CQcepstrum_temp, f_d),
            Deltas(Deltas(CQcepstrum_temp, f_d), f_d)
        ], axis=0)
    
    return CQcc.T