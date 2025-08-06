# A deep learning method for field boundaries delineation from remote sensing imagery with high boundary connectivity
The official PyTorch implementation of "A deep learning method for field boundary delineation from remote sensing imagery with high boundary connectivity".<br>


## Sample process
Experimental data for Agricultural Parcels can refer to our another repository  [The-collected-and-outlined-cropland-parcel](https://github.com/BNU-zhu/The-collected-and-outlined-cropland-parcel), which summarizes some of the data we have collected and outlined<br>  


## Requirements
`PyTorch  
TensorboardX  
GDAL  
OpenCV   
PIL  
numpy  
tqdm  
scikit-learn
accelerate
`  <br>

The code is tested under a Linux desktop with torch 1.2.0 and Python 3.6 <br>

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <train>
    ├── image
    |   ├── 001.tif
    │   ├── 002.tif
    │   ├── 003.tif
    │   ├── ...
    |
    └── region
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
    └── boundary
    |   ├── 001.tif
    |   ├── 002.tif
    |   ├── 003.tif
    |   ├── ...
```

For test datasets, the same structure as the above.<br>

## Run the model
First, modify the path in train.py, (note that --train_path should point to the image folder).<br>

Our image loading and model parameters are currently for 3-band images. <br>
If you use multi-band images, <br>
you need to modify the image normalization parameters in the `image_transform function` of `dataloader_2.py`, and specify the `input_channels parameter` for HCFNet.<br>

**When running train.py, there are two options**:<br>

**Directly run `python train.py`**.<br>

**Execute `accelerate launch train.py` to use Hugging Face Accelerate framework for accelerated training**.<br>
