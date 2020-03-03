import torch
import torchvision
import argparse
from model import Model
from dataset import transforms, load_one_image
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from scipy.stats import shapiro

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, help="Paths to the images as a glob expression (do not forget to escape asteriks)")
parser.add_argument("--age", type=int, default=-1, help="Age to mark in the plot")
parser.add_argument("--outfile", default="test.pdf", help="Path to the output file")
args = parser.parse_args()

# These are the model checkpoints used for the Analysis section of the paper.
# The best model can differ for new runs.
model_paths = [(0.0, "checkpoints_UTKFace/model_0.0_01.p"),
                (0.3, "checkpoints_UTKFace/model_0.3_05.p"),
                (0.8, "checkpoints_UTKFace/model_0.8_08.p"),
                (0.9, "checkpoints_UTKFace/model_0.9_08.p")]

number_of_classes = 90 # 61
device = "cuda" if torch.cuda.is_available() else "cpu"

models = []
for model_path in model_paths:
    model = Model(number_of_classes=number_of_classes)
    model.load_state_dict(torch.load(model_path[1], map_location=device))
    model.eval()
    models.append(model)

probs = [np.zeros((number_of_classes,)) for m in models]
predictions = [np.zeros((number_of_classes,)) for m in models]
real_ages = np.zeros((number_of_classes,))

images = glob(args.images)

for image_path in tqdm(images):
    age = int(image_path.split("/")[-1].split("_")[0])
    real_ages[age - 1] += 1

    image = load_one_image(image_path, transforms)

    for i in range(len(models)):
        output = models[i](image).data.numpy()[0]
        probs[i] += output
        predictions[i][np.argmax(output)] += 1


probs = [prob / len(images) for prob in probs]
real_ages = real_ages / len(images)

x = np.arange(1, number_of_classes+1)
plt.figure(figsize=(3.5, 2))
if args.age > 0:
    plt.axvline(x=args.age, c="#cccccc", linewidth=1.0)
else:
    plt.fill_between(x, real_ages, label="real ages", color="#cccccc")

for i, prob in enumerate(probs):
    plt.plot(x, prob, label="r = " + str(model_paths[i][0]), alpha=0.75, c=hsv_to_rgb([0.0, 1.0, model_paths[i][0]]), linewidth=0.7)

plt.legend()
plt.savefig(args.outfile)

os.system(f"open {args.outfile}")
