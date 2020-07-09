# Automated Dependence Plots
The following repository is the code for the following paper:

> [Automated Dependence Plots](http://www.auai.org/uai2020/proceedings/503_main_paper.pdf)<br/>
> David I. Inouye, Liu Leqi, Joon Sik Kim, Bryon Aragam, Pradeep Ravikumar<br/>
> [*Uncertainty in Artificial Intelligence* (UAI)](http://www.auai.org/uai2020/), 2020.

If you use this code, please cite this paper:
```
@inproceedings{inouye2020adp,
    author = {Inouye, David I and Leqi, Liu and Kim, Joon Sik and Aragam, Bryon and Ravikumar, Pradeep},
    booktitle = {Uncertainty in Artificial Intelligence},
    title = {{Automated Dependence Plots}},
    year = {2020}
}
```

## Introduction

> _How can we audit black-box machine learning models to detect undesirable behaviours?_ 

Visualizing the output of a model via dependence plots is a classical technique to understand how model predictions change as we vary the inputs. Automated dependence plots (ADPs) are a way to automate the manual selection of interesting or relevant dependence plots by optimizing over the space of dependence plots.

The basic idea is to define a utility function that quantifies how "interesting" or "relevant" a plot is---for example, this could be directions over which the model changes abruptly (`MostCurvatureUtility`), is non-monotonic (`LeastMonotonicUtility`), or oscillates (`TotalVariationUtility`). The steps are as follows:

1. Define a plot utility measure (or use a pre-defined utility)
2. Optimize over directions in feature space to find plots with the highest utility
3. Visualize the dependence plot in this direction

For example, the following figure highlights the combination of two features over which a model exhibits the most non-monotonic behaviour, and displays the output of the model as you vary these features:

<img width="500" src="https://user-images.githubusercontent.com/8812505/86979839-e3fbf300-c147-11ea-9c03-c0c8a630ef55.png" />

A more interesting example finds interesting directions in the latent space of a generative model (in this case, a VAE trained on the MNIST dataset):
<br/><br/>
<img width="800" src="https://user-images.githubusercontent.com/8812505/86980071-85834480-c148-11ea-85fa-04688b95c964.png" />


## Quickstart
To setup an environment (via conda), download data and pretrained models, and run notebooks to generate figures,
simply run the following commands:
```setup
make conda-env
source activate adp-env || conda activate adp-env
make data
make models
make test
```

## Examples

We have provided two Jupyter notebook tutorials to illustrate the use of ADPs:

- [A real data tutorial](https://github.com/davidinouye/automated-dependence-plots/blob/master/notebooks/demo-tutorial.ipynb) based on a UCI dataset
- [A toy example](https://github.com/davidinouye/automated-dependence-plots/blob/master/notebooks/demo-toy.ipynb) based on simulated data

These tutorials showcase the basic functionality and structure of the code. From here, users can extend these examples to custom plot utility measures and more complex datasets.

## Installation

### Requirements
We use a slightly older version of scikit-learn (0.19) but otherwise the packages are fairly standard.
To setup a conda environment and install requirements: 
```setup
make conda-env
conda activate adp-env
```
Or to do it manually:
```setup
conda env create -f environment.yml
conda activate adp-env
```
To remove this environment:
```setup
conda env remove --name adp-env
```

### Data and pretrained models setup
For simplicity, it's probably best to download all data and pretrained models before going through the tests.
The longest setup is for the GTSRB sign dataset but it should only take a few minutes.
Just run the following make commands to setup both the models and data
```setup
make models
make data
```

### Notebooks
Notebooks should be run by starting Jupyter notebooks in the notebooks folder.
This is to make sure the relative paths work correctly for loading the module and data/models.
```bash
cd notebooks/
jupyter notebook
```
### Figures
Each figure can be reproduced by running the following notebooks:

1. Figure 1 - figure-loan-optimize.ipynb
2. Figure 2 - figure-local-vs-counterfactual.ipynb 
3. Figure 3 - figure-lipschitz-bounded.ipynb
4. Figure 4 - figure-loan-model-comparison.ipynb
5. Figure 5 - figure-selection-bias.ipynb
6. Figure 6 - figure-streetsign.ipynb
7. Figure 7 - figure-vae-mnist.ipynb
8. One appendix figure - figure-domain-mismatch-loan.ipynb

### To run notebooks from command line
```bash
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute notebooks/NOTEBOOK_NAME.ipynb
```
We have provided a Makefile for running all the notebooks.  Merely run the following command to execute all notebooks (output goes intout notebooks/results/NOTEBOOKNAME.out.  An \*.error file will be generated if the notebook failed and a \*.success file will be generated if the notebook ran successfully.
```bash
make test
```

## Other Notes
The "counterfactual" link is to maintain backwards compatability with code developed for a previous paper manuscript. 
