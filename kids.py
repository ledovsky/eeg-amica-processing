import argparse
import sys
import warnings
from datetime import datetime, timezone
from os import listdir
from os.path import join

import mne
import numpy as np
import pandas as pd
from utils import save_montage
from mne.io import RawArray

from save_edf import write_mne_edf

CHANNELS = ['Fp1', 'Fpz', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8', 'FC3', 'FCz', 'FC4', 'FT7', 'FT8', 'C3', 'Cz', 'C4',
    'CP3', 'CPz', 'CP4', 'P3', 'Pz', 'P4', 'TP7', 'TP8', 'T3', 'T4', 'T5', 'T6', 'O1', 'Oz', 'O2']


def read_raw_csv(path: str) -> RawArray:
    """Reads csv file of the following structure
    rows - channels
    columns - time
    separator - space 
    no header
    sampling frequency = 500 Hz
    """
    raw_nums = pd.read_csv(path, header=None, sep=' ', engine='python')

    channels_to_use = CHANNELS
    sfreq = 500

    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    info = mne.io.meas_info.create_info(channels_to_use, sfreq=sfreq, ch_types="eeg")
    raw = RawArray(raw_nums.values, info)
    raw.set_montage(ten_twenty_montage)
    raw.set_meas_date(datetime.utcnow().replace(tzinfo=timezone.utc))

    return raw


def preprocess(raw_path: str, preprocessed_path: str, montage_path: str) -> None:
    """Takes raw csv and save them as edf files. Also saves the file with electrodes montage
    
    Args:
        raw_path: Input directory with raw csv. No other files are allowed
        preprocessed_path: Output directory for preprocessed EDF
        locations_path: Output path for locations file with filename
    """
    fns = listdir(raw_path)

    if len(fns) == 0:
        raise Exception("No raw files provided")

    for i, fn in enumerate(fns):
        path = join(raw_path, fn)
        raw = read_raw_csv(path)

        if i == 0:
            montage = raw.info.get_montage()
            save_montage(montage, montage_path)
    
        # Save edf
        edf_fn = fn.replace('.csv', '.edf')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            write_mne_edf(raw, join(preprocessed_path, edf_fn), overwrite=True)


def postprocess(preprocessed_path: str, amica_raw_path: str, postprocessed_path: str) -> None:
    fns = listdir(preprocessed_path)

    for fn in fns:
        print(fn)
        sample = mne.io.read_raw_edf(join(preprocessed_path, fn))

        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        sample.set_montage(ten_twenty_montage);

        ica = mne.preprocessing.ICA(n_components=len(CHANNELS))
        ica.info = sample.info
        ica.ch_names = sample.info["ch_names"]

        fn_s = fn.replace('.edf', '_amica_s.csv')
        fn_w = fn.replace('.edf', '_amica_w.csv')
        s = pd.read_csv(join(amica_raw_path, fn_s), header=None)
        w = pd.read_csv(join(amica_raw_path, fn_w), header=None)
        

        # consider to switch to mne/preprocessing/ica.py: read_ica_eeglab

        ica.unmixing_matrix_ = np.dot(w, s)
        icawinv = np.linalg.inv(np.dot(w, s))
        ica.mixing_matrix_ = icawinv
        ica.pca_components_ = np.eye(len(CHANNELS))
        ica.n_components_ = len(CHANNELS)
        ica._ica_names = [f'ICA{i:03d}' for i in range(len(CHANNELS))]
        ica.current_fit = 'eeglab_no_svd'
        ica.pre_whitener_ = np.ones([len(CHANNELS), 1])
        ica.pca_mean_ = np.zeros([len(CHANNELS), 1])
        ica.pca_explained_variance_ = np.zeros([len(CHANNELS), len(CHANNELS)])
        ica.reject_ = None

        fn_out = fn.replace('.edf', '_amica_ica.fif')
        ica.save(join(postprocessed_path, fn_out), overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.add_argument('--raw-path', type=str, required=True)
    preprocess_parser.add_argument('--preprocessed-path', type=str, required=True)
    preprocess_parser.add_argument('--montage-path', type=str, required=True)

    postprocess_parser = subparsers.add_parser('postprocess')
    postprocess_parser.add_argument('--preprocessed-path', type=str, required=True)
    postprocess_parser.add_argument('--amica-raw-path', type=str, required=True)
    postprocess_parser.add_argument('--postprocessed-path', type=str, required=True)

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess(raw_path=args.raw_path, preprocessed_path=args.preprocessed_path, montage_path=args.montage_path)
    elif args.command == 'postprocess':
        postprocess(preprocessed_path=args.preprocessed_path, amica_raw_path=args.amica_raw_path, postprocessed_path=args.postprocessed_path)
    else:
        parser.print_help()
        sys.exit(1)