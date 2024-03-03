# Image Similarity Search Using CLIP or ResNet

This repository contains a Python project for performing image similarity searches using either the CLIP or ResNet models. This tool allows users to extract features from an image dataset and then search within that dataset for images similar to a given query image.

## Requirements

- Python 3.6 or later
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- timm (PyTorch Image Models)
- numpy
- tqdm
- matplotlib

## Installation

1. Clone this repository to your local machine.
2. Ensure you have Python 3.6 or later installed.
3. Install the required Python packages using the following command:

```
pip install torch torchvision Pillow numpy tqdm matplotlib timm
```

## Usage

### Arguments

- `--input_size`: The size to which input images are resized. Default is 128.
- `--dataset_dir`: Directory path for the dataset from which features will be extracted.
- `--test_image_dir`: Directory path for the test images to search against the dataset.
- `--save_dir`: Directory path where the output images will be saved.
- `--model_name`: The model to use for feature extraction. Choices are 'resnet50', 'resnet152', or 'clip'. Default is 'resnet50'.
- `--feature_dic_file`: The name of the file where extracted image features will be saved. Default is 'corpus_feature_dict.npy'.
- `--topk`: The number of top similar images to retrieve. Default is 5.
- `--mode`: Operation mode. Choices are 'extract' to extract features or 'predict' to perform the image search. Default is 'predict'.

### Feature Extraction

First, extract features from your dataset by setting `--mode` to 'extract'. Ensure your dataset is structured so that images are sorted into subdirectories within the `--dataset_dir` path.

```shell
python image_similarity.py --mode extract --dataset_dir /path/to/dataset --model_name resnet50
```

### Image Search

To search for images similar to those in a specified directory, set `--mode` to 'predict'. The script will use the previously extracted features to find and display the top similar images according to the `--topk` argument.

```shell
python image_similarity.py --mode predict --test_image_dir /path/to/test_images --topk 5
```

### Output

The script saves a figure for each query image in the `--save_dir` path, showing the query image and its top-K similar images from the dataset. The figure also displays the similarity score and uses the model specified by `--model_name`.

## Note

- Make sure you have a compatible GPU for CUDA if you intend to use GPU acceleration with PyTorch. Otherwise, the script will default to using the CPU.
- The path arguments (`--dataset_dir`, `--test_image_dir`, `--save_dir`) must be absolute paths.
