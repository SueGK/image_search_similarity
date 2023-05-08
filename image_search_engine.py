import os
import argparse
import matplotlib.pyplot as plt
import glob
import sys
import tqdm
import numpy as np
import torch
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision
from PIL import Image
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args_parse():
    parser = argparse.ArgumentParser(description='Image search engine')
    parser.add_argument('--input_size', type=int, default=128, help='input image size')
    parser.add_argument('--dataset_dir', type=str, default='/Users/ggk/Downloads/dataset_fruit_veg/train', help='path to load images')
    parser.add_argument('--test_image_dir', type=str, default='/Users/ggk/Downloads/dataset_fruit_veg/val_images', help='images to test')
    parser.add_argument('--save_dir', type=str, default='/Users/ggk/Downloads/dataset_fruit_veg/output_dir', help='path to save images')
    parser.add_argument('--model_name', type=str, default='resnet50', help='model name') #resnet50, resnet152, clip
    parser.add_argument('--feature_dic_file', default='corpus_feature_dict.npy', help='image representation file')
    parser.add_argument('--topk', type=int, default=5, help='topk')
    parser.add_argument('--mode', type=str, default='predict', help='extract or predict')

    return parser.parse_args()


def extract_feature_single(args, model, file):
    img_rgb = Image.open(file).convert('RGB')
    image = img_rgb.resize((args.input_size, args.input_size), Image.ANTIALIAS)
    image = torchvision.transforms.ToTensor()(image)
    
    trainset_mean = [0.47083899, 0.43284143, 0.3242959]
    trainset_std = [0.37737389, 0.36130483, 0.34895992]
    image = torchvision.transforms.Normalize(trainset_mean, trainset_std)(image).unsqueeze(0).to(device) #todo remove device
   
    with torch.no_grad():
        features = model.forward_features(image)
        vec = model.global_pool(features)
        vec = vec.cpu().squeeze().numpy() #todo remove cpu

    img_rgb.close()

    return vec

    

def extract_feature_by_CLIP(model, preprocess, file):
    image = preprocess(Image.open(file)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec.cpu().squeeze().numpy()#todo remove cpu
    return vec


def extract_features(args, model, image_path="", preprocess=None):
    allVectors = {}
    for image_file in tqdm.tqdm(glob.glob(os.path.join(image_path, '*', '*.jpg'))):
        if args.model_name == "clip":
            allVectors[image_file] = extract_feature_by_CLIP(model, preprocess, image_file)
        else:
            allVectors[image_file] = extract_feature_single(args, model, image_file)
    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)
    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dic_file}", allVectors)
    return allVectors

def getSimilarityMatrix(vectors_dict):
    v = np.array(list(vectors_dict.values())) #[NUM, DIM]
    numerator = np.matmul(v, v.T) #[NUM, NUM]
    denominator = np.matmul(np.linalg.norm(v, axis=1, keepdims=True), np.linalg.norm(v, axis=1, keepdims=True).T) #[NUM, NUM]
    sim = numerator / denominator
    keys = list(vectors_dict.keys())
    return sim, keys

def setAxes(ax, image, query=False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize = 12)
        ax.xaxis.label.set_color('red')
    else:
        ax.set_xlabel("score={1:1.3f}\n{0}".format(image, value), fontsize = 12)
        ax.xaxis.label.set_color('blue')
    ax.set_xticks([])
    ax.set_yticks([])

def plotSimilarity(args, image, simImages, simValues, numRow=1, numCol=4):
    fig = plt.figure()

    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f'use engine model: {args.model_name}', fontsize=35)

    for j in range(0, numCol*numRow):
        ax = []
        if j == 0:
            img = Image.open(image)
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image.split(os.sep)[-1], query=True)
        else:
            img = Image.open(simImages[j-1])
            ax.append(fig.add_subplot(numRow, numCol, j+1))
            setAxes(ax[-1], simImages[j-1].split(os.sep)[-1], value = simValues[j-1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()

    fig.savefig(f"{args.save_dir}/{args.model_name}_search_top_{args.topk}_{image.split(os.sep)[-1].split('.')[0]}.png")
    plt.show()




#if __name__ == 'main':
model_names = timm.list_models(pretrained=True)
args = get_args_parse()

model = None
preprocess = None
#pdb.set_trace()
if args.model_name != 'clip':
    model = timm.create_model(args.model_name, pretrained=True)
    n_parameters = sum( p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters (M): %.2f' % (n_parameters / 1e6))
    model.eval()
else:
    # https://github.com/openai/CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

if args.mode == 'extract':
    # Stage 1: extract features
    print('extracting features...')
    print(f'Use pretrained model: {args.model_name}')
    allVectors = extract_features(args, model, image_path=args.dataset_dir, preprocess=preprocess)

else:
    # Stage 2: image search
    print('image search...')
    print(f'use pretrained model {args.model_name} to search {args.topk} similar images from corpus')
    
    test_images = glob.glob(os.path.join(args.test_image_dir, "*.png"))
    test_images += glob.glob(os.path.join(args.test_image_dir, "*.jpg"))
    test_images += glob.glob(os.path.join(args.test_image_dir, "*.jpeg"))

    # loading image representation dictory
    allVectors = np.load(f"{args.save_dir}/{args.model_name}/{args.feature_dic_file}", allow_pickle=True)
    allVectors = allVectors.item()

    # reading test images
    for image_file in tqdm.tqdm(test_images):
        print(f'reading {image_file}')
        if args.model_name == 'clip':
            # CLIP model
            allVectors[image_file] = extract_feature_by_CLIP(model, preprocess, image_file)
        else:
            # Resnet50/152 model
            allVectors[image_file] = extract_feature_single(args, model, image_file)

    sim, keys = getSimilarityMatrix(allVectors)
    
    results = {}

    for image_file in tqdm.tqdm(test_images):
        print(f"'sorting most similar images as {image_file} . . .")
        index = keys.index(image_file)
        sim_vec = sim[index]

        indexs = np.argsort(sim_vec)[::-1][1:args.topk] # start from 1 because 0 is the original image
        
        simImages, simScores = [], []

        for idx in indexs:
            simImages.append(keys[idx])
            simScores.append(sim_vec[idx])
        results[image_file] = (simImages, simScores)

    print('starting to show simlar images...')
    for image_file in test_images:
        # ----------------------------- simImages,              simValues----------------
        plotSimilarity(args, image_file, results[image_file][0], results[image_file][1], numRow=1, numCol=args.topk)



