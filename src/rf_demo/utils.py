import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import orthogonal_procrustes

def plot_ground_truth_cortex_rf(monkey, data_dir='../../demo/data', title_prefix=None):
    """
    Load and plot ground truth cortical and RF data for a monkey using standard colors (colorsWAlpha).
    Plots both maps side by side as in main figures.
    Args:
        monkey (str): e.g. 'monkey_l' or 'monkey_a'
        data_dir (str): path to demo data
        title_prefix (str): optional prefix for subplot titles
    """
    cortex_file = f'{data_dir}/demo_{monkey}_cortex_gt.npz'
    rf_file = f'{data_dir}/demo_{monkey}_rf_gt.npz'
    colors_file = f'{data_dir}/demo_{monkey}_colors_gt.npy'
    try:
        cortex_data = np.load(cortex_file)
        rf_data = np.load(rf_file)
        colorsWAlpha = np.load(colors_file)
    except Exception as e:
        print(f"Error loading ground truth files: {e}")
        return
    cortex_mm = cortex_data['mm']
    rf_mm = rf_data['mm']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Cortex plot
    axes[0].scatter(cortex_mm[:, 0], cortex_mm[:, 1], c=colorsWAlpha[:, :3], s=40, edgecolor='black', alpha=0.8)
    axes[0].set_title(f'{monkey} Ground Truth Cortex')
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    axes[0].set_aspect('equal')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    # RF plot
    axes[1].scatter(rf_mm[:, 0], rf_mm[:, 1], c=colorsWAlpha[:, :3], s=40, edgecolor='black', alpha=0.8)
    axes[1].set_title(f'{monkey} Ground Truth RF')
    axes[1].set_xlabel('X (deg)')
    axes[1].set_ylabel('Y (deg)')
    axes[1].set_aspect('equal')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    fig.suptitle(f'{title_prefix+" " if title_prefix else ""}{monkey} Ground Truth Maps', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all_bands_seismic(monkey, bands=None, duration=0.5, data_dir='../../demo/data', sampling_rate=1000, title_prefix=None):
    """
    Plot all available bands for a monkey in two rows using the seismic colormap.
    Args:
        monkey (str): e.g. 'monkey_l' or 'monkey_a'
        bands (list): list of bands to plot (default: all)
        duration (float): seconds to plot
        data_dir (str): path to demo data
        sampling_rate (int): Hz
        title_prefix (str): optional prefix for subplot titles
    """
    if bands is None:
        bands = ['lfp', 'low', 'alpha', 'beta', 'gamma', 'highgamma']
    samples = int(sampling_rate * duration)
    time = np.linspace(0, duration, samples)
    n_bands = len(bands)
    fig, axes = plt.subplots(2, (n_bands + 1) // 2, figsize=(16, 6), squeeze=False)
    for i, band in enumerate(bands):
        filename = os.path.join(data_dir, f'demo_{monkey}_{band}_15s.npz')
        if not os.path.exists(filename):
            axes[i // ((n_bands + 1) // 2), i % ((n_bands + 1) // 2)].set_visible(False)
            continue
        data = np.load(filename)
        lfp = data['lfp']
        lfp_segment = lfp[:, :samples]
        ax = axes[i // ((n_bands + 1) // 2), i % ((n_bands + 1) // 2)]
        im = ax.imshow(lfp_segment, aspect='auto', cmap='seismic', extent=[time[0], time[-1], 0, lfp_segment.shape[0]])
        ax.set_title(f'{title_prefix+" " if title_prefix else ""}{monkey} {band}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_seismic_data(data, time=None, title=None, xlabel='Time (s)', ylabel='Channel'):
    """
    Plot a 2D array (channels x time) using the seismic colormap.
    Args:
        data: 2D numpy array (channels x time)
        time: 1D array for time axis (optional)
        title: plot title (optional)
        xlabel, ylabel: axis labels
    """
    plt.figure(figsize=(10, 4))
    if time is not None:
        extent = (time[0], time[-1], 0, data.shape[0])
        plt.imshow(data, aspect='auto', cmap='seismic', interpolation='nearest', extent=extent, origin='lower')
    else:
        plt.imshow(data, aspect='auto', cmap='seismic', interpolation='nearest', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ---- Vendored minimal math helpers (from top-level code/utils.py) ----

def calculate_corr_of_distances(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between pairwise Euclidean distance matrices of two point sets."""
    D = euclidean_distances(a)
    U = euclidean_distances(b)
    cm = np.corrcoef(D.ravel(), U.ravel())
    return float(cm[0, 1])


def scipy_Antonio_procrustes(data1: np.ndarray, data2: np.ndarray):
    """Procrustes returning (mtx1, mtx2, disparity, R, s, norm1, norm2, mean1, mean2).

    Matches the behavior relied upon by the demo pipeline.
    """
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    mean1 = np.mean(mtx1, 0)
    mean2 = np.mean(mtx2, 0)
    mtx1 -= mean1
    mtx2 -= mean2
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")
    mtx1 /= norm1
    mtx2 /= norm2
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    disparity = np.sum(np.square(mtx1 - mtx2))
    return mtx1, mtx2, disparity, R, s, norm1, norm2, mean1, mean2


def rescale_procrustes_map(H: np.ndarray, R: np.ndarray, s: float, norm1: float, norm2: float,
                           mean1: np.ndarray, mean2: np.ndarray) -> np.ndarray:
    """Rescale aligned map back to the original reference scale (utah map scale)."""
    proH = (np.dot(((H - mean2) / norm2), R.T) * s)
    recoveredH = (proH * norm1) + mean1
    return recoveredH
