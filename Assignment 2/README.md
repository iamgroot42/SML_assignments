## Assignment 1

### Running it
  - Download relevant data using `wget http://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data`.
  - Install relevant libraries : `pip install -r requirements.txt`.
  - For Q4, run `python3 statistics.py`.
  - For Q5, run `python3 main.py` to generate data splits. Use `python3 train_bayes.py tae.data 1` for 5-fold cross validation, and `python3 train_bayes.py tae.data 0` to train on 70% of data.
  - Use `python3 test_bayes.py test.data model` to log performance on test data using the model saved above.
