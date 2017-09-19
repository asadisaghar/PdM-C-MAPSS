#### Installation

sudo apt-get install libfreetype6-dev libfreetype6 libblas-dev libblas3 libblas-dev liblapack-dev libatlas-base-dev gfortran
pip install -r requirements.txt


#### Data

https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#turbofan

#### Preprocessing

- preprocessing.ipynb (parameters to set)
  - dependency: preprocessing_tools.py

#### Classification
- unsupervised_classification.ipynb (parameters to set)
- supervised_classification.ipynb (parameters to set)
  - dependency: classification_tools.py
- LSTM_classification.ipynb (parameters to set)
  - dependency: LSTM_tools.py