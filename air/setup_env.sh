pip install --upgrade setuptools
pip install --upgrade pip
virtualenv env && \
source my_env/bin/activate && \
pip install -r requirements.txt --allow-all-external
