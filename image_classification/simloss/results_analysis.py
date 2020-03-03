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

    r = re.compile(r"Run:\s*(\d+)\nLower bound:\s*(\d+\.\d+)\nBest epoch:\s*(\d+)\nBest evaluation metrics:\s*accuracy:\s*(\d+\.\d+),\s*superclass_accuracy:\s*(\d+\.\d+),\s*failed_superclass_accuracy:\s*(\d+\.\d+),\s*loss:\s*(\d+\.\d+)\nBest test metrics:\s*accuracy:\s*(\d+\.\d+),\s*superclass_accuracy:\s*(\d+\.\d+),\s*failed_superclass_accuracy:\s*(\d+\.\d+),\s*loss:\s*(\d+\.\d+)")

    m = r.search(text)

    info = {
        "run": m[1],
        "lower_bound": m[2],
        "epoch": m[3],
        "val_accuracy": m[4],
        "val_sa": m[5],
        "val_fsa": m[6],
        "val_loss": m[7],
        "test_accuracy": m[8],
        "test_sa": m[9],
        "test_fsa": m[10],
        "test_loss": m[11]
    }

    results.append(info)

df = pd.DataFrame(results)

df.to_csv(dir + "/results.csv")
