# **Software Description**

# Overview

This is the software script for implementing the ROI method published in XXX, as well as reproducing the results described in the manuscript. This software contains four major components:

- The data preprocessing interface for TCGA dataset.
- The performance of time to event prediction from four different models: CPH, CPH\_ROI, CPH\_DL, and CPH\_DL\_ROI.
- The evaluation of performance of four modes using a CVD dataset collected by the national health heart study.
- Use a synthetic dataset to study the performance of four different models.

# Software Structure

The examples folder contains the scripts for reproducing the result in Fig 1.

The data folder contains the script file to ensemble dataset from TCGA cohort.

The model folder contains the files for deep Cox neural network (CPH\_DL), Cox regression with ROI (CPH\_ROI), and Cox regression with deep learning and ROI (CPH\_DL\_ROI).

The simulation folder contains two synthetic datasets simulated from a uniform distribution, a data sampler, and two files, for reproducing the results in Table 3.

| **Entity** | **Path/location** | **Note** |
| --- | --- | --- |
| Cox | ./model/cox\_model.py | The Cox regression model |
| FeatureExtractor | ./model/cox\_model.py | The Feature extractor for deep learning models |
| CPH\_ROI | ./model/cox\_model.py | The CPH\_ROI model as described in the method section of our manuscript. |
| CPH\_DL | ./model/cox\_model.py | The CPH\_DL model as described in the method section of our manuscript. |
| CPH\_DL\_ROI | ./model/cox\_model.py | The CPH\_DL\_ROI model as described in the method section of our manuscript. |
| Synthetic datasets 1, 2 | ./simulation/main.csv./simulation/related.csv | The synthetic datasets for simulation study. |
| SimulatedData | ./simulation/simulate\_data.py | The class for synthetic data generation. |
| summary | ./results/summary.py | To summarize the result of each experiment. |

# System Requirements

## Software dependency

The system relies on the following software, reagent, or resources.

| **REAGENT or RESOURCE** | **SOURCE** | **IDENTIFIER** |
| --- | --- | --- |
| TCGA | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| TCGA Cancer Types | Broad Institute | [https://gdac.broadinstitute.org/](https://gdac.broadinstitute.org/) |
| American Cancer Types | Cancer Treatment Centers of America | [https://www.cancercenter.com/cancer-types](https://www.cancercenter.com/cancer-types) |
| TCGA Protein | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| TCGA Clinical | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| TCGA Clinical Endpoints | TCGA Pan-Cancer Clinical Data Resource | [https://www.sciencedirect.com/science/article/pii/S0092867418302290](https://www.sciencedirect.com/science/article/pii/S0092867418302290) |

## Software version

Our software has been tested on the following software version.

| **Software and Hardware** | **SOURCE** | **IDENTIFIER** |
| --- | --- | --- |
| REAGENT or RESOURCE | SOURCE | IDENTIFIER |
| torch 1.7.1 | PyTorch Enterprise | https://pytorch.org/ |
| Python 3.7 | Python Software Foundation | [https://www.python.org/download/releases/2.7/](https://www.python.org/download/releases/2.7/) |
| Computational Facility | The National Institute for Computational Sciences | [https://www.nics.tennessee.edu/computing-resources/acf](https://www.nics.tennessee.edu/computing-resources/acf) |
| Numpy 1.15.4 | Tidelift, Inc | https://libraries.io/pypi/numpy/1.15.4 |
| Numpydoc 0.9.1 | Tidelift, Inc | [https://libraries.io/pypi/numpydoc](https://libraries.io/pypi/numpydoc) |
| Scipy 1.2.1 | The SciPy community | [https://docs.scipy.org/doc/scipy-1.2.1/reference/](https://docs.scipy.org/doc/scipy-1.2.1/reference/) |
| Seaborn 0.9.0 | Michael Waskom | [https://seaborn.pydata.org/installing.html](https://seaborn.pydata.org/installing.html) |
| Sklearn 0.0 | The Python community | [https://pypi.org/project/sklearn/](https://pypi.org/project/sklearn/) |
| Skrebate 0.6 | Tidelift, Inc | [https://libraries.io/pypi/skrebate](https://libraries.io/pypi/skrebate) |
| Keras 2.2.4 | GitHub, Inc. | [https://github.com/keras-team/keras/releases/tag/2.2.4](https://github.com/keras-team/keras/releases/tag/2.2.4) |
| Keras-Applications 1.0.8 | GitHub, Inc. | [https://github.com/keras-team/keras-applications](https://github.com/keras-team/keras-applications) |
| Keras-Preprocessing 1.1.0 | GitHub, Inc. | [https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0](https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0) |
| Tensorboard 1.13.1 | GitHub, Inc. | [https://github.com/tensorflow/tensorboard/releases/tag/1.13.1](https://github.com/tensorflow/tensorboard/releases/tag/1.13.1) |
| Tensorflow 1.13.1 | tensorflow.org | [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) |
| Tensorflow-estimator 1.13.1 | The Python community | [https://pypi.org/project/tensorflow-estimator/](https://pypi.org/project/tensorflow-estimator/) |
| Statsmodels 0.9.0 | Statsmodels.org | [https://www.statsmodels.org/stable/release/version0.9.html](https://www.statsmodels.org/stable/release/version0.9.html) |
| Lifelines 0.16.3 | Cam Davidson-Pilon Revision | [https://lifelines.readthedocs.io/en/latest/Changelog.html](https://lifelines.readthedocs.io/en/latest/Changelog.html) |
| Xlrd 1.2.0 | The Python community | [https://pypi.org/project/xlrd/](https://pypi.org/project/xlrd/) |
| XlsxWriter 1.1.8 | The Python community | [https://pypi.org/project/XlsxWriter/](https://pypi.org/project/XlsxWriter/) |
| Xlwings 0.15.8 | The Python community | [https://pypi.org/project/xlwings/](https://pypi.org/project/xlwings/) |
| Xlwt 1.3.0 | The Python community | [https://pypi.org/project/xlwt/](https://pypi.org/project/xlwt/) |

## Hardware requirements

We recommend use a GPU (V100) to speed up the running process of our software.

# Installation Guide

Our software package can be downloaded from the following github page: [https://github.com/AtlasGao/ROI](https://github.com/AtlasGao/ROI). This package contains the source code and demo datasets to reproduce the results represented in our paper. Our software can run on Windows and Ubuntu, but we suggest using Linux system which is easier for environment configuration.

Conda –install requirements.txt

Requirements.txt

numpy==1.15.4

numpydoc==0.9.1

scipy==1.2.1

seaborn==0.9.0

sklearn==0.0

skrebate==0.6

torch==1.7.1

Keras==2.2.4

Keras-Applications==1.0.8

Keras-Preprocessing==1.1.0

tensorboard==1.13.1

tensorflow==1.13.1

tensorflow-estimator==1.13.0

statsmodels==0.9.0

lifelines==0.16.3

Optunity==1.1.1

xlrd==1.2.0

XlsxWriter==1.1.8

xlwings==0.15.8

xlwt==1.3.0

# Demo

## Instructions to run on data

The python scripts used to generate the Figure 3 in our paper can be found in the following folder

cd /ROI/examples

python tcga\_test.py

The python scripts used to generate the Table 4 in our paper can be found in the following folder

cd /ROI/simulation

python run\_simulation.py

After the execution, the result will be printed in the console.

## Expected output

The output of each task will be a vector with 4 C-index values, show the time to event prediction performance from four different models: CPH, CPH\_ROI, CPH\_DL, CPH\_DL\_ROI.

# Instructions for Use

## How to run the software

To run our software with different diseases, endpoints, or feature, you need to download our dataset from a shared location (probably 10 GB) and put it under the ROI/data/ folder. You can simply specify the task you want to run by passing the location of your interested dataset to the _read\_data_ function.

## Reproduction instructions

The key point to reproduce the result in our paper is follow the configuration process strictly.

## Authors

Yan Gao and Yan Cui, ({ygao45, ycui2}@uthsc.edu).

## License

This project is covered under the GNU General Public License (GPL).
