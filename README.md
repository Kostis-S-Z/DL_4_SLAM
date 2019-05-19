# Deep Learning for Second Language Acquisition Modeling

Using RNNs for Second Language Acquisition Modeling based on the 2018 competition by Duolingo http://sharedtask.duolingo.com/

### Requirements

Python 3.5/7 and other libraries. To install everything needed please run

```
pip install requirements.txt
```

### Execute

1. *Unpack data*

In the data folder give permissions to the unpacking script by running:

```
chmod 777 unpack_data.sh
```

then execute it to unpack the data in put it in a folder.

```
./unpack_data.sh
```

In the starter_code folder, unpack the glove.6B.50d.txt.tar.gz that contains pretrained word embeddings:

```
tar -xvzf glove.6B.50d.txt.tar.gz
```


2. Run count_features.py to go over all data and make a dictionary of the features and how many times each value appears (important for feature selection)

```
python3 count_features.py
```

3. Execute either lstm.py for a default run of a model or experiments.py to run different comparative experiments.

```
python3 lstm.py
```

or run baseline.py for Duolingo's baseline Logistic Regression model.
