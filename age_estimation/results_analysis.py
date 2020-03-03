from glob import glob
import pandas as pd
import re
from tqdm import tqdm
import sys

dir = sys.argv[1]

results = []

files = glob(dir + "/*.out")

for file in tqdm(files):
    with open(file) as f:
        text = "".join(f.readlines())

    r = re.compile(r"RUN:\s*(\d+)\nREDUCTIONFACTOR:\s*(\d+\.\d+)\nBEST EPOCH:\s*(\d+)\nBEST VALIDATION:\s*accuracy: (\d+\.\d+), MAE: (\d+\.\d+), MSE: (\d+\.\d+), loss: (\d+\.\d+)\nBEST TEST:\s*accuracy: (\d+\.\d+), MAE: (\d+\.\d+), MSE: (\d+\.\d+), loss: (\d+\.\d+)")

    m = r.search(text)

    info = {
        "run": m[1],
        "reduction_factor": m[2],
        "epoch": m[3],
        "val_accuracy": m[4],
        "val_MAE": m[5],
        "val_MSE": m[6],
        "val_loss": m[7],
        "test_accuracy": m[8],
        "test_MAE": m[9],
        "test_MSE": m[10],
        "test_loss": m[11]
    }

    results.append(info)

df = pd.DataFrame(results)

df.to_csv(dir + "/results.csv")
