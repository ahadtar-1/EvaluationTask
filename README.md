# EvaluationTask

The project implements RNN Transducer for the purpose of automatically converting audio speech to text.

# Set up and Installation

### Case 1 - Run Project

In order to run the project, a conda environment should be created (**python 3.7**) to preserve the existant packages and dependencies in the system. The requirements file should then be run in the environment to import the required dependencies for the project. Once packages are imported the project can be run.

```bash
conda create --name py37 python=3.7

source activate py37

pip install -r requirements.txt
```

```bash
python3 stream.py --flagfile ./flagfiles/E6D2 LARGE Batch.txt  --name rnnt-m-bpe  --model_name english_43_medium.pt  --step_n_frame 2
```
