import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import os
import scipy.stats as scp
from scipy.signal import hilbert
import itertools
import timecorr as tc
from numba import prange


def lm_parallel(DV, IV, covar):
    """
    Perform linear model with covariates
    
    """
    X = np.hstack([IV, covar, np.ones((covar.shape[0],1))])
    beta = np.zeros((X.shape[1], DV.shape[1]))
    resid = np.zeros(DV.shape)
                   
    for i in prange(0,DV.shape[1]):
        y = DV[:,i]
        inv_mat = np.linalg.pinv(X)
        beta[:,i] = np.dot(inv_mat,y)
        yhat=np.sum(np.transpose(beta[:,i])*X,axis=1)
        resid[:,i] = y - np.transpose(yhat)
    return(beta[:-covar.shape[1],:], resid)


def compile_ts_data(subdf, movie, datadir, outfile):
    """
    combine timeseries data for each movie together into 1 file
    
    Parameters
    ----------
    subdf: DataFrame
        A dataframe with subject IDs as the index. Includes IDs for all usable data.
    movie: str
        Corresponds with the str for the movie content to concatenate (e.g., "DM" or "TP").
    datadir: folder path
        Path to folder with the subject timeseries ciftis.
    outfile: file path
        Path including filename to save the output data of shape Ntimepoints x Nparcels x Nsubjects.
    
    Returns
    -------
    data: numpy array
        The compiled data of shape Ntimepoints x Nparcels x Nsubjects
    """
    if not isinstance(subdf, pd.DataFrame):
        subdf = pd.read_csv(subdf, index_col=0)
    
    for sub in subdf.index:
        file = '{0}{1}_task-movie{2}_bold1_AP_Atlas_rescale_resid0.9_filt_gordonseitzman.32k_fs_LR.ptseries.nii'.format(datadir,sub, movie)
        if sub == subdf.index[0]:
            data = StandardScaler().fit_transform(nib.load(file).get_fdata())
            data = np.expand_dims(data, axis=2)
        else:
            t = StandardScaler().fit_transform(nib.load(file).get_fdata())
            t = np.expand_dims(t, axis=2)
            data = np.concatenate([data,t],axis=2)
    
    print('Compile data from {0} brain regions measured at {1} timepoints from {2} participants.'.format(data.shape[1],data.shape[0],data.shape[2]))
    np.save(outfile, data)
    return(data)


def compute_phase(group_ts_data, outfile):
    """
    compute phase angles for each parcel timeseries
    
    Parameters
    ----------
    group_ts_data: filepath OR numpy array
        File or numpy array with compiled timeseries data of shape Ntimepoints x Nparcels x Nsubjects 
        OR Ntimepoints x Nfeatures
    
    Returns
    -------
    phase_data: numpy array
        The compiled data of shape Ntimepoints x Nparcels x Nsubjects
    """
    
    if not isinstance(group_ts_data, np.ndarray):
        group_ts_data = np.load(group_ts_data)
    
    phase_data = np.zeros_like(group_ts_data)
    
    if len(group_ts_data.shape)==3:
        for b in range(0,group_ts_data.shape[2]):
            group_ts_data[:,:,b] = StandardScaler().fit_transform(group_ts_data[:,:,b])
            for a in range(0,group_ts_data.shape[1]):
                phase_data[:,a,b] = np.angle(hilbert(group_ts_data[:,a,b]), deg=False)
    elif len(group_ts_data.shape)==2:
        group_ts_data = StandardScaler().fit_transform(group_ts_data)
        for a in range(0,group_ts_data.shape[1]):
                phase_data[:,a] = np.angle(hilbert(group_ts_data[:,a]), deg=False)
    
    np.save(outfile, phase_data)
    
    return(phase_data)



def compute_ips(group_phase_data, outprefix, intersub=True, interregion=False, savemean=True):
    """
    parcel-wise instantaneous phase synchrony- output pairwise IPS and mean global IPS
    
    Parameters
    ----------
    group_phase_data: numpy array
        The compiled data of shape Ntimepoints x Nparcels x Nsubjects
    outprefix: string
        The filepath and file prefix for the saved IPS data.
    intersub: bool
        Set to True to compute intersubject phase synchrony.  Set to False for inter-region.
    interregion: bool
        Set to True to computer inter-region instantaneous phase synchrony. Set to False for intersubject. 
    savemean: bool
        Set to True to save average IPS (across subjects)
        
    Returns
    -------
    ips_data: numpy array
        Instantaneous phase synchrony data of shape Nparcels x Nsubjects x Nsubjects x Ntimepoints 
        OR Nsubjects x Nparcels x Nparcels x Ntimepoints
    mean_ips_data: numpy array
        Instantaneous phase synchrony data, averaged across time, of shape Nparcels x Ntimepoints
        
    """
    
    if not isinstance(group_phase_data, np.ndarray):
        group_phase_data = np.load(group_phase_data)
    
        
    if intersub:
        if os.path.isdir(outprefix):
            file_name = os.path.join(outprefix, 'ips_data.dat')
        else:
            file_name = outprefix + 'ips_data.dat'
        ips_data = np.memmap(file_name, dtype=np.float32, mode='w+',
                              shape=(group_phase_data.shape[1],
                                     group_phase_data.shape[2],
                                     group_phase_data.shape[2],
                                     group_phase_data.shape[0]))

        subs = range(0, group_phase_data.shape[2])
        for region in range(0, group_phase_data.shape[1]):
            combs = itertools.combinations(subs, 2)
            for c in combs:
                sub1 = group_phase_data[:, region, c[0]]
                sub2 = group_phase_data[:, region, c[1]]
                a = 1 - np.sin(np.abs(sub1 - sub2) / 2)
                ips_data[region,c[0],c[1],:] = a
                ips_data[region,c[1],c[0],:] = a

        if savemean:
            mask = np.tri(ips_data.shape[2], ips_data.shape[2], -1, dtype=int)
            mean_ips_data = np.mean(ips_data[:,mask==1,:], axis=1)
            if os.path.isdir(outprefix):
                mean_file_name = os.path.join(outprefix, 'mean_isps_data.npy')
            else:
                mean_file_name = outprefix + 'mean_isps_data.npy'
            np.save(mean_file_name, mean_ips_data.T)
            return(mean_ips_data, ips_data)
        else:
            return(ips_data)
        
    if interregion:
        if os.path.isdir(outprefix):
            file_name = os.path.join(outprefix, 'ips_data.npy')
        else:
            file_name = outprefix + 'ips_data.npy'
        ips_data = np.empty((group_phase_data.shape[2],
                             group_phase_data.shape[1],
                             group_phase_data.shape[1],
                             group_phase_data.shape[0]))

        regions = range(0, group_phase_data.shape[1])
        for sub in range(0, group_phase_data.shape[2]):
            combs = itertools.combinations(regions, 2)
            for c in combs:
                sub1 = group_phase_data[:, c[0], sub]
                sub2 = group_phase_data[:, c[1], sub]
                a = 1 - np.sin(np.abs(sub1 - sub2) / 2)
                ips_data[sub,c[0],c[1],:] = a
                ips_data[sub,c[1],c[0],:] = a
        np.save(file_name, ips_data)
        
        if savemean:
            mean_ips_data = np.mean(ips_data, axis=0)
            if os.path.isdir(outprefix):
                mean_file_name = os.path.join(outprefix, 'mean_isps_data.npy')
            else:
                mean_file_name = outprefix + 'mean_isps_data.npy'

            np.save(mean_file_name, mean_ips_data.T)
            return(mean_ips_data, ips_data)
        else:
            return(ips_data)

        
def temporal_smooth(data, time, sampling_rate, window=4):
    """
    Parameters
    ----------
    data: numpy array
        1-D array with signal data to smooth.
    time: numpy array
        Time stamps in seconds for the signals to be smoothed.
    sampling_rate: float
        The sampling rate in Hz that the data were acquired in.
    window: int
        The size of the gaussian kernel to use for smoothing (must be even number).
    
    Returns
    -------
    smoothed: numpy array
        1-D array with smoothed data.
    
    """
    def gaussian(t, fwhm):
        return np.exp(-(4*np.log(2)*t**2)/fwhm**2)

    # create kernel
    n = len(time)
    k = int(window/2)
    gtime = np.arange(-k, k)/sampling_rate

    gauswin = gaussian(gtime, window)
    gauswin = gauswin/np.sum(gauswin)
    
    # zeropad the data
    pad_data = np.pad(data, (window,window), mode='constant', constant_values=0)
    
    # smooth data
    smoothed = np.zeros_like(pad_data)
    for i in range(k+1, n-k-1):
        smoothed[i] = np.sum(pad_data[i-k:i+k] * gauswin)
    # remove pad
    smoothed = smoothed[window:-window]
    return(smoothed)

        
def brain_bx_crosscorr(brain, bx):
    """
    
    Parameters
    ----------
    brain: numpy array
        neural phase data of shape Nsubjects x Nparcels x Nparcels x Ntimepoints
    bx: numpy array
        Nfeatures x Ntimepoints
    
    Returns
    -------
    cross_corr: numpy array
        
    
    """
    
    cross_corr = np.empty([brain.shape[0], brain.shape[1], brain.shape[2], bx.shape[1]])

    # compute lags
    for sub in range(0, brain.shape[0]):
        for n1 in range(0, brain.shape[1]):
            for n2 in range(0, brain.shape[1]):
                if n1!=n2:
                    for b in range(0, bx.shape[1]):
                        res = correlate(brain[sub, n1, n2, :], bx[:,b], mode='same')
                        lags = correlation_lags(brain[sub, n1, n2, :].shape[0], bx[:,b].shape[0], mode='same')
                        cross_corr[sub, n1, n2, b] = lags[np.argmax(res)]
                        cross_corr[sub, n2, n1, b] = lags[np.argmax(res)]
    
    return(cross_corr)
    
    
def compute_dFC(ts, dFCflat_file):
    """
    Computes the dynamic connectivity between regions using a gaussian weighted sliding window (10 second windows).
    
    Parameters
    ----------
    ts: numpy array
        Participant timeseries in the shape of Ntimepoints x Nregions.
    dFCflat_file: file path
        The filename to save the flattened dynamic connectivity data.
    dFCmat_file: file path
        The filename to save the matrix dynamic connectivity data.
    """
    # run first participant to get measuresments
    dFC_flat = tc.timecorr(ts, weights_function=tc.gaussian_weights, 
                            weights_params={'var': 10}, combine=tc.corrmean_combine)
    np.save(dFCflat_file, dFC_flat)