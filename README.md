
This repository is the official PyTorch implementation of "LSTC-MDA: A Unified Framework for Long-Short Term Temporal Convolution and Mixed Data Augmentation in Skeleton-Based Action Recognition". LSTC-MDA achieves state-of-the-art performance in skeleton-based action recognition.

## Requirements
> - Python >= 3.9.18
> - PyTorch >= 2.6.0
> - Platforms: Ubuntu 22.04, CUDA 12.2
> - We have included a dependency file for our experimental environment. To install all dependencies, create a new Anaconda virtual environment and execute the provided file. Run `conda env create -f environment.yml`.
> - Run `pip install -e torchlight`.

## Data Preparation

### Download datasets

#### There are 3 datasets to download:

- NTU RGB+D
- NTU RGB+D 120
- NW-UCLA

#### NTU RGB+D and NTU RGB+D 120

1. Request dataset from [here](https://rose1.ntu.edu.sg/dataset/actionRecognition)
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

- Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```
## Training
```bash
cd LSTC-MDA

# Train LSTC-MDA on NTU RGB+D X-Sub60 dataset (joint modality)
python main_jmda_addmix.py --config ./config/train/ntu_cs/SkateFormer_j_tconv_jmda_add_mixup.yaml

# Train LSTC-MDA on NTU RGB+D X-Sub60 dataset (bone modality)
python main_jmda_addmix.py --config ./config/train/ntu_cs/SkateFormer_b_tconv_jmda_add_mixup.yaml

# Train LSTC-MDA on NTU X-View60 dataset (joint modality)
python main_jmda_addmix_cv.py --config ./config/train/ntu_cv/SkateFormer_j_tconv_jmda_add_mixup.yaml

# Train LSTC-MDA on NTU 120 X-Set120 dataset (joint modality)
python main_jmda_addmix_cv.py --config ./config/train/ntu120_cset/SkateFormer_j_tconv_jmda_add_mixup.yaml 

# Train LSTC-MDA on NW-UCLA dataset (joint modality)
python main_jmda_addmix_ucla.py --config ./config/train/nw_ucla/SkateFormer_j_tconv_jmda_add_mixup.yaml
```
