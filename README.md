# Multi Task Crystal Graph Convolutional Neural Networks (MT-CGCNN)

This repository implements the Multi Task Crystal Graph Convolutional Neural Networks (MT-CGCNN) introduced in our paper titled "[MT-CGCNN: Integrating Crystal Graph Convolutional Neural Network with Multitask Learning for Material Property Prediction](https://arxiv.org/abs/1811.05660)". The model that takes as input a crystal structure and predicts multiple material properties in a multi-task setup.

The package provides code to train a MT-CGCNN model with a customized dataset. This is built on an existing model [CGCNN](https://github.com/txie-93/cgcnn) which the authors suggest to checkout as well.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a MT-CGCNN model](#train-a-cgcnn-model)
- [License](#license)
- [Citation](#cite)

##  Prerequisites

The package requirements are listed in `requirements.txt` file. Run the below code to install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Define a customized dataset 

To define a customized dataset, you need a list of CIFs for which you want to train the model.
In this work, we use data from [Materials Project](https://www.materialsproject.org/). To generate the data for training this model using Materials Project, you may use the file `utils.py`. In this file, you need to provide the API key to access Materials Project (available after logging into the website), the folder where you wish to save the data, the list of mpids (Materials Project IDs) for which data needs to be saved and a list of properties for which the model has to be trained. To reproduce the results for this paper, you should use the `mpids_full.csv` as the list of mpids for which data is downloaded. Also, you may define your own custom dataset for training the model.

A customized dataset stored in a folder `root_dir` will have the following files:

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with multiple columns. The first column recodes a unique `ID` for each crystal. From second column onwards the value of respective target property is stored. For eg., if you wish to perform multi-task learning for `Formation energy` and `Band gap`, then the second column should have the target value for `Formation energy` of the crystal and thrid column should have the target value for `Band gap`.

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

One example of customized dataset is provided in the repository at `data/sample/`. This contains `100` samples with `Formation energy` and `Band gap` as target properties

### Train a MT-CGCNN model

To see training of an instance of MT-CGCNN, you can just run the following:

```bash
python init.py
```
This will run a demo using datasource: `data/sample/` and some pre-defined set of parameters. But, you can change and play with the parameters of the model using the `init.py` file. To run multiple iterations of the same experiment (one motivation can be to get average error), you can run the following code:

```bash
python init.py --idx 1
```

To reproduce results stated in the paper, you might need to tune the parameters in the hyperparameter space mentioned in the paper. Also, average MAE is reported for 5 runs of the experiment.

After training, you will get multiple files in `results` folder present within the datasource (For eg., for demo case results will be saved in `data/sample/results/0/`). The most important ones are:

- `model_best.pth.tar`: stores the MT-CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the MT-CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target values, and predicted values for each crystal in test set.
- `logfile.log`: A complete log of the experiment (useful for DEBUGGING purposes)

The other files are useful to understand how well-trained the model is and can be referred for DEBUGGING purposes. Briefly, the files contain information about the training & validation losses, training & validation errors and also some useful plots.

## License

MT-CGCNN is released under the MIT License.

## Citation
Please consider citing our paper if you use this code in your work
```
@article{sanyal2018mt,
  title={MT-CGCNN: Integrating Crystal Graph Convolutional Neural Network with Multitask Learning for Material Property Prediction},
  author={Sanyal, Soumya and Balachandran, Janakiraman and Yadati, Naganand and Kumar, Abhishek and Rajagopalan, Padmini and Sanyal, Suchismita and Talukdar, Partha},
  journal={arXiv preprint arXiv:1811.05660},
  year={2018}
}
```
