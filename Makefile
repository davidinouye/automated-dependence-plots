# A way to run various quick scripts
# A bit hacky way to run all the ipython notebooks and put the files into notebooks/results
ALL_NOTEBOOKS=$(shell ls notebooks/ | grep ".ipynb" | sed 's/\.ipynb/\.success/g' | sed 's/^/notebooks\/results\//g')
MKDIR_P=mkdir -p

.PHONY: all data models test conda-env clean clean-conda-env

all: 
	echo "No default make, 'make conda-env' to create conda environment, 'make data' to download data, 'make models' to download pretrained models, and 'make test' to test all Jupyter notebooks."

conda-env:
	-conda env create -f environment.yml
	echo -e '\nUse the following command to activate the conda environment\n\nsource activate adp-env || conda activate adp-env'

clean-conda-env:
	-conda env remove --name adp-env
	echo -e '\nUse the following command to deactivate the conda environment\n\nconda deactivate || source deactivate'

test: $(ALL_NOTEBOOKS)

simpletest: notebooks/results/a-error-test.success
	
imagetest: notebooks/results/figure-streetsign.success notebooks/results/figure-vae-mnist.success

notebooks/%.success: notebooks/results
	# Runs jupyter notebook and only adds the file if it successfully runs
	-jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute notebooks/$(shell basename $@ | sed 's/[.]success//g').ipynb --output-dir=notebooks/results &> ${basename $@}.out && touch $@ || touch ${basename $@}.error

notebooks/results:
	${MKDIR_P} notebooks/results


models: models/mnist_cnn.pt models/traffic_model.h5 models/vae_mnist.pt

models/%:
	# Download zip and extract into models folder
	-mkdir models
	curl -L https://app.box.com/shared/static/lieaf3cdna8krkhmjlpyakp6n3wkpclz.zip --output adp-example-models.zip
	unzip -o adp-example-models.zip
	rm adp-example-models.zip

data: data/MNIST data/Data_1980.csv data/german.data data/GTSRB 

data/MNIST:
	-mkdir data;
	python notebooks/helpers.py

data/Data_1980.csv:
	-mkdir data;
	python notebooks/helpers.py

data/german.data:
	-mkdir data;
	python notebooks/helpers.py

data/GTSRB:
	-mkdir data;
	# Download raw train data
	curl https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip --output data/GTSRB_Final_Training_Images.zip;
	unzip data/GTSRB_Final_Training_Images.zip -d data/GTSRB_Final_Training_Images;
	rm data/GTSRB_Final_Training_Images.zip;
	# Download raw test data
	curl https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip --output data/GTSRB_Final_Test_Images.zip;
	unzip data/GTSRB_Final_Test_Images.zip -d data/GTSRB_Final_Test_Images;
	rm data/GTSRB_Final_Test_Images.zip
	# Download annotations 
	curl https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip --output data/GTSRB_Final_Test_GT.zip;
	unzip data/GTSRB_Final_Test_GT.zip -d data;
	rm data/GTSRB_Final_Test_GT.zip;
	# Run preprocessing for GTSRB
	python scripts/data_utils_sign.py

clean:
	rm -r notebooks/results
	rm -r models/cached*.pkl

