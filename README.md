# DS1011-FinalProject
This is our final project for course DS-GA 1011 Natural Language Processing with Representation Learning.

In this project we implement two models, the bilateral multi-perspective matching (BiMPM) model and the decomposable attention (DECATT) model to the paraphrase identification task on the Quora question pair dataset. The codes of the original models we implemented are collected from https://www.kaggle.com/lamdang/dl-models (DECATT) and https://github.com/ijinmao/BiMPM_keras (BiMPM). Both are implemented on Keras with a tensorflow backend.


## Requirements
- python 2.7
- tensorflow (1.2.1)
- keras (2.1.2)
- numpy
- scipy
- gensim
- tqdm
- pattern
- nltk
- pandas
- scikit-learn
- h5py

To get Glove pre-trained embeddings:

$ wget http://nlp.stanford.edu/data/glove.840B.300d.zip

Then unzip the file and save at the 'embeddings' folder.

For training within each model folder, just run

> python train_model.py

For parameters tuning, in BiMPM: change model configuration in config.py; in DECATT: change model parameters in models/decom_attn.py.

The test predictions will be generated after finishing the training process, saved in the 'subm' folder as a .csv file, where each test example is given a probability indicating whether the two questions are identical.
