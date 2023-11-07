from os import listdir
from os.path import join
from tempfile import TemporaryDirectory

import mne
import numpy as np
import pandas as pd

from kids import postprocess, preprocess, read_raw_csv


def test_read_raw_csv():
    input_path = join('test_data', 'kids', 'raw', 'test_1.csv')
    raw = read_raw_csv(input_path)
    assert raw.info['nchan'] == 31

    # produce test_data
    np.random.seed(42)
    pd.DataFrame(np.random.rand(31, 1000) * 1e-6).to_csv('test_data/kids/raw/test_1.csv', header=False, index=False, sep=' ')
    pd.DataFrame(np.random.rand(31, 1000) * 1e-6).to_csv('test_data/kids/raw/test_2.csv', header=False, index=False, sep=' ')


def test_preprocess():
    raw_path = join('test_data', 'kids', 'raw')
    with TemporaryDirectory() as preprocessed_path, TemporaryDirectory() as montage_dir:
        montage_path = join(montage_dir, 'montage.loc')
        preprocess(raw_path, preprocessed_path, montage_path)

        assert len(listdir(preprocessed_path)) == 2
        
        raw = mne.io.read_raw_edf(join(preprocessed_path, 'test_1.edf'))
        assert raw.info['nchan'] == 31

    # produce test_data
    preprocessed_path = join('test_data', 'kids', 'preprocessed')
    montage_path = join('test_data', 'kids', 'montage.loc')
    preprocess(raw_path, preprocessed_path, montage_path)


def test_postprocess():
    preprocessed_path = join('test_data', 'kids', 'preprocessed')
    amica_raw_path = join('test_data', 'kids', 'amica_raw')

    with TemporaryDirectory() as postprocessed_path:
        postprocess(preprocessed_path, amica_raw_path, postprocessed_path)
        assert len(listdir(postprocessed_path)) == 2
        ica = mne.preprocessing.read_ica(join(postprocessed_path, 'test_1_amica_ica.fif'))
        assert ica.info['nchan'] == 31

    # produce test_data
    postprocessed_path = join('test_data', 'kids', 'postprocessed')
    postprocess(preprocessed_path, amica_raw_path, postprocessed_path)