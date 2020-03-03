# Age Estimation Experiment

Reproducing our Age Estimation experiments involves the following steps:

* Download and extract the AFAD and UTKFace datasets into the `data` folder (i.e. `data/AFAD` and `data/UTKFace`)
* Remove the images of over 90 year old persons from UTKFace
* Organize all images into folders that are named by the age class with length three (e.g. `025` or `003`)
* Execute `run.sh`, which will test the reduction factors from the paper on both datasets
* After all runs have finished, produce the results as csv files with `python3 results_analysis.py results_AFAD` and `python3 results_analysis.py results_UTKFace`
* The resulting files are `results_AFAD/results.csv` and `results_UTKFace/results.csv`

## Evaluating images

We provide the pretrained model weights that we used in the Analysis section of the paper. They can be found in `checkpoints_UTKFace`.
The plots from the paper can then be generated using the `evaluate_one_image.py` script.