# Self-supervised learning of the relationships between speech sounds, articulatory gestures and phonetic units

## Project organization

* `communicative_agent/`
  * Full agent
* `extras/`
  * Extras scripts used for optional verifications
* `imitative_agent/`
  * Agent without discretization mechanisms (uses acoustic space as goal)
* `out/`
  * Contains scripts output (trained models, cached results)
* `synthesizer/`
  * The articulatory synthesizer
* `datasets/`
  * Contains the imported and processed datasets
* `external/`
  * Contains the imported resources (raw datasets, LPCynet)
* `lib/`
  * Scripts and libraries used by the whole project
* `quantizer/`
  * Acoustic and articulatory quantizers
* `datasets_infos.yaml`
  * File describing the datasets to import
* `README.md`
  * This file
* `preprocess_datasets.py`
  * Script used for datasets importation and processing
* `features_config.yaml`
  * Describes the data format used globally
* `phones_infos.yaml`
  * Describes the phonetic content of the datasets
* `requirements.txt`
  * Versions of Python and libraries used

## Setup

### LPCNet importation

1. Download and compile the LPCyNet wrapper for LPCNet from https://github.com/georgesma/lpcynet.
2. Place the resulting files in `./external/lpcynet`.

### Datasets importation

To import a dataset, specify its content in the `datasets_infos.yaml` file and run `python preprocess_datasets.py`.

#### Predefined datasets

Some datasets are already predefined in the `datasets_infos.yaml` configuration file.

##### PB2007

Download `PB2007.zip` from https://zenodo.org/record/6390598 and extract its content in `./raw_datasets/pb2007`.

##### MOCHA-TIMIT

Download `fsew0_v1.1.tar.gz` and `msak0_v1.1.tar.gz` from https://data.cstr.ed.ac.uk/mocha/ and extract their content in `./external/raw_datasets/fsew0`  and `./raw_datasets/msak0` respectively.

#### `datasets_infos.yaml` format

To import a custom dataset, use the following description format:

```yaml
# Dataset name
pb2007:
    # [Required]
    # Path to wav files (`glob` format)
    wav_pathname: ./external/raw_datasets/pb2007/_wav16/*.wav

    # [Optional]
    # Name of the other dataset used as a reference for normalizing the RMS of sounds in the current dataset
    # Note: the other dataset must be specified before the current one
    wav_rms_reference: other_dataset_name

    # [Optional] If the dataset contains EMA data
    # Path to EMA data files (`glob` format)
    ema_pathname: ./external/raw_datasets/pb2007/_seq/*.seq
    # EMA files format (`seq` or `est`)
    ema_format: seq
    # EMA sampling rate (in hertz)
    ema_sampling_rate: 100
    # Factor by which the EMA coordinates must be divided to be in millimeters
    ema_scaling_factor: 0.1
    # Column indices corresponding to the coordinates:
    # [lower_incisor_x, lower_incisor_y, tongue_tip_x, tongue_tip_y, tongue_middle_x, tongue_middle_y, tongue_back_x, tongue_back_y, lower_lip_x, lower_lip_y, upper_lip_x, upper_lip_y, velum_x, velum_y]
    # (velum coordinates are optional)
    ema_coils_order: [0, 6, 1, 7, 2, 8, 3, 9, 5, 11, 4, 10]
    # Set to true to apply a lowpass filter to EMA coordinates before importation
    ema_needs_lowpass: false

    # [Optional] If the dataset contains labelisation data
    # Path to the `.lab` files (`glob` format)
    lab_pathname: ./external/raw_datasets/pb2007/_lab/*.lab
    # Factor by which the `.lab` timings must be divided to be in seconds
    lab_resolution: 10000000

    # [Optional] Name of the phonemic information group related to the dataset described in `phones_infos.yaml`
    # No need to specify it if the group has the same name as the dataset
    phones_infos: mocha
```
