# Self-supervised speech unit discovery from articulatory and acoustic features using VQ-VAE

## Setup

### LPCNet importation

1. Download and compile the LPCyNet wrapper for LPCNet from https://github.com/georgesma/lpcynet.
2. Place the resulting files in `./external/lpcynet`.

### Datasets importation

1. Download `PB2007.zip` from https://zenodo.org/record/6390598 and extract it in `./raw_datasets/pb2007`.
2. Download `fsew0_v1.1.tar.gz` and `msak0_v1.1.tar.gz` from https://data.cstr.ed.ac.uk/mocha/ and extract them in `./external/raw_datasets/fsew0`  and `./raw_datasets/msak0` respectively.
3. Run `python preprocess_datasets.py` to extract the features needed for the experiments.
