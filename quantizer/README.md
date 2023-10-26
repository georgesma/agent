# Quantizer

The quantizer is a VQ-VAE trained either from acoustic data (`cepstrum`), articulatory data (`art_params`) or both modalities combined.

## `optimize_hyperparameters.py`

This script is used to optimize the quantizer's hyperparameters in relation to its global ABX score.

```
python quantizer/optimize_hyperparameters.py
```

It starts from the configuration described in `quantizer_config.yaml` and explores the hyperparameters space described in the script itself.

Once optimization is complete, use the notebook `visu_abx.ipynb` to explore the results.

## `train.py`

This script is used to train `NB_TRAINING` (by default 5) versions of the quantizer.

Once training is complete, use the notebook `visu_figures.ipynb` to explore the results.
