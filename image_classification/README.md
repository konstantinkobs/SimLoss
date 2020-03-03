# Image Classification Experiment

Reproducing our Image Classification experiments involves the following steps:

* Install this package with `pip3 install -r requirements.txt`
* Build the similarity matrix with `python3 simloss/data/make_sim_matrix.py` (however, the resulting file can already be found in `./data/processed`)
* Run the experiment with `run.sh` within the `simloss` folder
* After all runs have finished, produce the results as csv files with `python3 simloss/results_analysis.py simloss/results`
* The resulting file is `results/results.csv`
