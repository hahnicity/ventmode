# ventmode

Reproducible, Accurate, Ventilator Mode Classifications using PB-840 Ventilator Data

## Citing
If you used my dataset or the work herein please cite us :bowtie:

```
@article{rehm2019improving,
  title={Improving Mechanical Ventilator Clinical Decision Support Systems with a Machine Learning Classifier for Determining Ventilator Mode.},
  author={Rehm, Gregory B and Kuhn, Brooks T and Nguyen, Jimmy and Anderson, Nicholas R and Chuah, Chen Nee and Adams, Jason Yeates},
  journal={Studies in health technology and informatics},
  volume={264},
  pages={318--322},
  year={2019}
}
```

## Install

Clone the repo

    git clone https://github.com/hahnicity/ventmode

You should use anaconda for installing dependencies. [Install anaconda](https://docs.anaconda.com/anaconda/install/) then create a new environment.

    conda create -n ventmode python=3.8
    source activate ventmode

There are a few packages must be installed using anaconda:

    conda install matplotlib seaborn numpy scipy pandas scikit-learn cython

If your system support pytorch and you want to try using the LSTM code then you will need
to install pytorch

    conda install pytorch

Install dtw libs and go back to ventmode

    git clone https://github.com/lukauskas/dtwco
    cd dtwco
    pip install -e .
    cd ../ventmode

Then install the rest via pip

    python setup.py develop

## Reproducing Work
You can reproduce the work we've done by running a bash script.

    cd ventmode
    ./experiments.sh

## Using This In Your Own Projects
We've designed this to be as easy as possible to use in your own projects. There is
an API that is capable of being used and there is also a CLI script. Let's start with the CLI.
First you will need to save an instance of the trained Random Forest model, and the data scaler. We recommend that you train your model on a 20% split of the data because we
have found this leads to best subsequent model performance.

    cd ventmode
    python main.py --save-classifier-to saved-ventmode-model.pkl -sr .2 --split-type simple

Then you can use the CLI by

    python scan_file.py <your filename> saved-ventmode-model.pkl saved-ventmode-model.pkl.scaler

For API usage:

```python
    from ventmode import datasets
    from ventmode.main import run_dataset_with_classifier

    fileset = {'x': [('patient1', <patient 1's file>), ('patient 2', <patient 2's file), ...]}
    vfinal = datasets.VFinalFeatureSet(fileset, 10, 100)
    df = vfinal.create_prediction_df()
    cls = pickle.load(open(<path to saved classifier>))
    scaler = pickle.load(open(<path to saved scaler>))
    model_results = run_dataset_with_classifier(cls, scaler, df, "vfinal")
```
