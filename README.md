# SimLoss: Class Similarities in Cross Entropy
> Konstantin Kobs, Michael Steininger, Albin Zehe, Florian Lautenschlager, and Andreas Hotho

This repository contains the code for the paper "SimLoss: Class Similarities in Cross Entropy" that was accepted as a short paper at [ISMIS 2020](https://ismis.ist.tugraz.at/2019/09/17/hello-world/):

> One common loss function in neural network classificationtasks is Categorical Cross Entropy (CCE), which punishes all misclassifications equally.
> However, classes often have an inherent structure.
> For instance, classifying an image of a rose as “violet” is better than as “truck”.
> We introduce SimLoss, a drop-in replacement for CCE that incorporates class similarities along with two techniques to  construct such matrices from task-specific knowledge.
> We test SimLoss on Age Estimation and Image Classification and find that it brings significant improvements over CCE on several metrics.
> SimLoss therefore allows for explicit modeling of background knowledge by simply exchanging the loss function, while keeping the neural network architecture the same.

The code for the Age Estimation and Image Classification experiments can be found in the folders `age_estimation` and `image_classification`, respectively.
Each folder contains a `README.md` file, which describes how to run that specific experiment.

## More Resources

To provide more insight into SimLoss, we also provide a version of this paper with an appendix on arXiv.
It explores the relation of SimLoss and other loss functions that use a probability matrix.

Link: **Coming Soon**

## Citation

```
Coming soon...
```
