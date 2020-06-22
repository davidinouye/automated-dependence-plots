# Automated Dependence Plots
The following repository is the code for the following paper.  If you use this code, please cite this paper:

> Automated Dependence Plots<br/>
> David I. Inouye, Liu Leqi, Joon Sik Kim, Bryon Aragam, Pradeep Ravikumar<br/>
> To appear in *Uncertainty in Artificial Intelligence* (UAI), 2020.

# Quickstart
To setup environment (via conda), download data and pretrained models, and run notebooks to generate figures,
simply run the following commands:
```setup
make conda-env
source activate adp-env || conda activate adp-env
make data
make models
make test
```

# Requirements
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

# Data and pretrained models setup
For simplicity, it's probably best to download all data and pretrained models before going through the tests.
The longest setup is for the GTSRB sign dataset but it should only take a few minutes.
Just run the following make commands to setup both the models and data
```setup
make models
make data
```

# Notebooks
Notebooks should be run by starting Jupyter notebooks in the notebooks folder.
This is to make sure the relative paths work correctly for loading the module and data/models.
```bash
cd notebooks/
jupyter notebook
```
## Figures
Each figure can be reproduced by running the following notebooks:

1. Figure 1 - figure-loan-optimize.ipynb
2. Figure 2 - figure-local-vs-counterfactual.ipynb 
3. Figure 3 - figure-lipschitz-bounded.ipynb
4. Figure 4 - figure-loan-model-comparison.ipynb
5. Figure 5 - figure-selection-bias.ipynb
6. Figure 6 - figure-streetsign.ipynb
7. Figure 7 - figure-vae-mnist.ipynb
8. One appendix figure - figure-domain-mismatch-loan.ipynb

## To run notebooks from command line
```bash
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute notebooks/NOTEBOOK_NAME.ipynb
```
We have provided a Makefile for running all the notebooks.  Merely run the following command to execute all notebooks (output goes intout notebooks/results/NOTEBOOKNAME.out.  An \*.error file will be generated if the notebook failed and a \*.success file will be generated if the notebook ran successfully.
```bash
make test
```

# Other Notes
The "counterfactual" link is to maintain backwards compatability with code developed for a previous paper manuscript. 
