# EEG Amica processing repo

## Dataset kids

Test data generation

```python
pd.DataFrame(np.random.rand(31, 1000) * 1e-6).to_csv('test_data/kids/raw/test_1.csv', header=False, index=False, sep=' ')
```

## About reading EEGLAB ICA

I used my custom creation of ICA objects. In the process of work under this project, I found the function that can be used to achieve almost the same in the original MNE code (```preprocessing/ica.py:read_ica_eeglab```). At first, I found that my code of matrix conversion was right (cool!). At second, I found the code which reconstructs the PCA/ICA decomposition which is used only in MNE. It was the only significant difference and I left my own code because I'm not sure that we need the described reconstruction.