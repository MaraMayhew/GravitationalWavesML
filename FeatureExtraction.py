import numpy as np
from pycbc.types import TimeSeries
import matplotlib.pyplot as plt
import os
from wdm_wavelet.wdm import WDM
import pandas as pd 

folder_path = "data_subset"

#Functions for loading in data 

def load_g2net_as_timeseries(path, sampling_rate=2048): # 2048 is the sampling rate of the kaggle simulated data
    """
    Loads a G2Net .npy file of shape (3, N) and returns
    PyCBC TimeSeries objects for each detector.
    
    Returns:
        H, L, V   (TimeSeries objects)
    """
    # Load the raw numpy file
    data = np.load(path)      # shape = (3, N)
    
    # Ensure expected format
    if data.ndim != 2:
        raise ValueError("Expected .npy file with shape (3, N).")

    detectors = []
    delta_t = 1.0 / sampling_rate

    for i in range(data.shape[0]):
        strain = data[i]
        ts = TimeSeries(strain, delta_t=delta_t)
        detectors.append(ts)

    # return H, L, V
    return detectors[0], detectors[1], detectors[2]



def load_g2net_file_as_timeseries(path, sampling_rate=2048):
    """
    Load a single G2Net .npy file and convert its detectors into
    PyCBC TimeSeries objects.
    Returns a dict: {"H": ts, "L": ts, "V": ts}
    """
    data = np.load(path)

    delta_t = 1.0 / sampling_rate
    detectors = {}

    # Mapping for readability
    label_map = {0: "H", 1: "L", 2: "V"}

    for i in range(data.shape[0]):
        ts = TimeSeries(data[i], delta_t=delta_t)
        detectors[label_map[i]] = ts

    return detectors


def load_g2net_folder(folder, sampling_rate=2048):
    """
    Loads every .npy file in a folder and returns:
        {
            "file_id": {"H": TimeSeries, "L": TimeSeries, "V": TimeSeries},
            ...
        }
    """
    file_dict = {}

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".npy"):
            continue

        fpath = os.path.join(folder, fname)

        # Extract file ID (strip .npy extension)
        file_id = fname.replace(".npy", "")

        try:
            detectors = load_g2net_file_as_timeseries(fpath, sampling_rate)
            file_dict[file_id] = detectors
            #print(f"Loaded {fname}")

        except Exception as e:
            print(f"ERROR loading {fname}: {e}")

    return file_dict

#Functions for feature extraction

import numpy as np

def nf_norm(matrix):
    """
    feature corresponding to size of matrix (norm). 
    """
    return np.linalg.norm(matrix) # Frobenius norm (sqrt of sum of squares)


def correlation_coefficient(sig1, sig2, sig3):
    """
    feature corresponsing to cross-correlation coefficient (Pearson coefficient) between two 1D signals.
    detector coherence 
    """
    s1 = sig1 - np.mean(sig1)
    s2 = sig2 - np.mean(sig2)
    s3 = sig3 - np.mean(sig3)

    num = np.sum(s1 * s2 * s3)

    den = np.sqrt(np.sum(s1**2) * np.sum(s2**2) * np.sum(s3**2))

    if den == 0:
        return 0.0
    return num / den


def chi2_reconstruction(residual, ndof):
    """
    chi^2 reconstruction quality stat
    residual noise/incoherent energy (En) per degree of freedom (Ndf)
    reconstruction is the reconstructed signal after the pipeline extracts a cluster of excess power and determines that its coherent across detectors

    residual: array, residual noise energy (En) - the residual waveform after subtracting reconstruction
    ndof: int, number of degrees of freedom, number of independent wavelet amplitudes describing the event
    Returns normalized chi2 = En / Ndf
    """
    En = np.sum(np.abs(residual)**2)
    return En / float(ndof)


def effective_correlated_snr(Ec, chi2):
    """
    feature: η0 = sqrt(Ec) / max(1, chi2), equation 1 in paper
    signal to noise ratio of combined event from multiple detectors

    Ec: coherent energy (float)
    chi2: chi2 reconstruction statistic (float)
    Returns: float
    """
    return np.sqrt(Ec) / max(1.0, chi2)


def snr_over_likelihood_ratio(snr, likelihood):
    """
    feature - SNRi / L
    
    snr: float (signal-to-noise ratio in a detector or network)
    likelihood: float (likelihood value from signal reconstruction)
    Returns: (snr^2) / likelihood ratio
    """
    if likelihood <= 0: # likelihood can't be negative
        return np.nan
    return (snr**2) / likelihood # but how to compute likelihood? 

def get_tfmap(H, L, V):
    wdm = WDM(32, 64, 6, 10)
    tf_mapH = wdm.t2w(H)
    tf_mapL = wdm.t2w(L)
    tf_mapV = wdm.t2w(V)
    return(tf_mapL, tf_mapV, tf_mapH) 
    
def get_event_pixels(tfmap, threshold=3.0):
    mag = np.abs(tfmap.data)
    return mag > threshold * np.std(mag)


def get_en_ec(H, L, V):
    '''
    # W is the reconstructed (coherent) amplitude of the signal in time–frequency pixel k
    # these are each of W's components from each detector, together they construct the full W vector
    # the coherent projection is projection of this W vector onto the subspace spanned by the expectted signal responses hx  and h+ which you get through the likelihood
    '''
    H=H.data
    L=L.data
    V=V.data
    mask = get_event_pixels(H) | get_event_pixels(L) # get all the event pixels from H or L

    total_energy = (np.sum(np.abs(H[mask])**2) + np.sum(np.abs(L[mask])**2) + np.sum(np.abs(V[mask])**2))

    Wh = np.sum(H[mask].real) 
    Wl = np.sum(L[mask].real)
    Wv = np.sum(V[mask].real)

    Ec = (Wh + Wl + Wv)**2
    En = total_energy - Ec

    if En < 0: # failsafe to prevent small numerical negatives ?
        En = 0   

    dof = np.sum(mask)
    return Ec, En, dof

#Creating the features to the data and saving to a CSV 

gw_data = load_g2net_folder(folder_path)
df_targets=pd.read_csv("labels_subset.csv")
df_targets=df_targets.set_index('id') 
targets=[]
norm=[]
cross_correlation=[]
chi2=[]
correlated_snr=[]
snr_likelihood=[]



for i in gw_data.keys():
    H = gw_data[i]["H"]
    L = gw_data[i]["L"]
    V = gw_data[i]["V"]
    mapH, mapL, mapV= get_tfmap(H,L, V) 
    H = (H.data - H.data.mean()) / H.data.std()
    L = (L.data - L.data.mean()) / L.data.std()
    V = (V.data - V.data.mean()) / V.data.std()

    targets.append(df_targets.loc[i, "target"])
    norm.append((nf_norm(H)+nf_norm(L)+nf_norm(V))/3)
    cross_correlation.append((correlation_coefficient(np.array(H),np.array(L),np.array(V))))

    Ec, En, dof= get_en_ec(mapH, mapL, mapV)
    chi2_val=chi2_reconstruction(En,dof)
    chi2.append(chi2_val) 
    correlated_snr_val=effective_correlated_snr(Ec, chi2_val)
    correlated_snr.append(correlated_snr_val)
    snr_likelihood.append(snr_over_likelihood_ratio(correlated_snr_val, Ec)) 

features = pd.DataFrame({
    
    "norm": norm,
    "cross_correlation": cross_correlation,
    "correlated_snr": correlated_snr,
    "chi2": chi2,
    "snr_likelihood": snr_likelihood
}, index=list(gw_data.keys()))  # set id as index

targets = pd.DataFrame({
    "target": targets
}, index=list(gw_data.keys()))

features.to_csv("features.csv")  # 

targets.to_csv("targets.csv")






