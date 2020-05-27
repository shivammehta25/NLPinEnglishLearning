# Application of Natural Language Processing in English Grammar Exercises

In this directory we will perform experimentation to deal with two learning problems:
* Reading Comprehension Exercise Generation
* Fill-in-the-Blank Exercise Generation


## Reading Comprehension Exercise Generation

Detailed Implementation can be found in Sequence2SequenceQuestionGenerator.ipynb notebook

### Baseline Model

```zsh
cd Sequence_2_sequence_Generation/Baseline
```

#### Preprocessing
```zsh
python preprocessing.py --help
```
Options
```
preprocessing.py [-h] [-d DATASET] [-m MODE] [-f]

Utility to Preprocess datasets currently available datasets: SQUAD

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Name of Dataset
  -m MODE, --mode MODE  Split on ANSWER or QUESTION
  -f, --filter          filter the sentences on answers
```
**Run:**
```zsh
python preprocessing.py
```

#### Training
```zsh
python train.py --help
```
Options
```
usage: train.py [-h] [-d DATASET] [-m MODEL] [-c CLIPNORM] [-l LEARNINGRATE]
                [-v] [-e EPOCHS] [-t TEACHERFORCING] [-tmp TRAINED_MODEL_PATH]

Utility to Train datasets {1: 'VanillaSeq2Seq'}

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        which dataset to train on
  -m MODEL, --model MODEL
                        Which Model to Train
  -c CLIPNORM, --clipnorm CLIPNORM
                        Value to clip gradients
  -l LEARNINGRATE, --learningrate LEARNINGRATE
                        Learning rate of Adam Optmizer
  -v, --validation      Flag to turn validation on and off
  -e EPOCHS, --epochs EPOCHS
                        Number of Epochs to train
  -t TEACHERFORCING, --teacherforcing TEACHERFORCING
                        Teacher Forcing
  -tmp TRAINED_MODEL_PATH, --trained-model-path TRAINED_MODEL_PATH
                        Load the model from the directory
```

### Sequence To Sequence Models

```zsh
cd Sequence_2_sequence_Generation/FairSeq_models
```
#### Preprocessing
```zsh
python preprocess.py
```
Options
```
usage: preprocess.py [-h] [-l LOCATION]

Utility to preprocess the dataset

optional arguments:
  -h, --help            show this help message and exit
  -l LOCATION, --location LOCATION
                        Location of Dataset if left empty configuration will
                        be used
```
#### Training
```zsh
python train.py --help
```
Options
```
usage: train.py [-h] [-m {LSTM,CNN,Transformer}] [-n NUM_EPOCHS]
                [-b BATCH_SIZE]

Utility to train the models

optional arguments:
  -h, --help            show this help message and exit
  -m {LSTM,CNN,Transformer}, --model {LSTM,CNN,Transformer}
                        Select the Seq2Seq Model to train
  -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Number of epochs to train
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Training Batch Size
```

#### Generate
```zsh
python generate.py --help
```
Options
```
usage: generate.py [-h] [-m {LSTM,CNN}] [-sm {best,last}] [-b BATCH_SIZE]

Utility to Generate Sentences from Test Set

optional arguments:
  -h, --help            show this help message and exit
  -m {LSTM,CNN}, --model {LSTM,CNN}
                        Select the Seq2Seq Model to train
  -sm {best,last}, --sub-model {best,last}
                        Select which model to generate with the one with best
                        valid loss or the last epoch trained model
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Training Batch Size
```

## Fill-in-the-Blank Exercise Generation

### Classification

**Machine Learning Classifier**

Detailed commands can be found in MachineLearningClassifiers.ipynb notebook

```zsh
cd classifier
```

**Deep Learning Classifier**

Detailed commands can be found in DeepLearningClassifiers.ipynb notebook

```zsh
python preprocessdata.py --help
```
Options
```
usage: preprocessdata.py [-h] [-l LOCATION]

Utility to preprocess the dataset

optional arguments:
  -h, --help            show this help message and exit
  -l LOCATION, --location LOCATION
                        Location of Dataset if left empty configuration will
                        be used
```

**Train**
```zsh
python train.py --help
```
Options:
```
usage: train.py [-h] [-s SEED] [-loc MODEL_LOCATION] [-b BIDIRECTIONAL]
                [-d DROPOUT] [-e EMBEDDING_DIM] [-hd HIDDEN_DIM] [-l N_LAYERS]
                [-lr LEARNING_RATE] [-n EPOCHS] [-batch BATCH_SIZE]
                [-f FREEZE_EMBEDDINGS] [-t {multi,answeronly}]
                [-l2 L2_REGULARIZATION]
                [-m {RNNHiddenClassifier,RNNMaxpoolClassifier,RNNFieldClassifier,CNN2dClassifier,CNN1dClassifier,RNNFieldClassifer,CNN1dExtraLayerClassifier}]
                [-lhd LINEAR_HIDDEN_DIM]

Utility to train the Model

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  Set custom seed for reproducibility
  -loc MODEL_LOCATION, --model-location MODEL_LOCATION
                        Give an already trained model location to use and
                        train more epochs on it
  -b BIDIRECTIONAL, --bidirectional BIDIRECTIONAL
                        Makes the model Bidirectional
  -d DROPOUT, --dropout DROPOUT
                        Dropout count for the model
  -e EMBEDDING_DIM, --embedding-dim EMBEDDING_DIM
                        Embedding Dimensions
  -hd HIDDEN_DIM, --hidden-dim HIDDEN_DIM
                        Hidden dimensions of the RNN
  -l N_LAYERS, --n-layers N_LAYERS
                        Number of layers in RNN
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate of Adam Optimizer
  -n EPOCHS, --epochs EPOCHS
                        Number of Epochs to train model
  -batch BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of Epochs to train model
  -f FREEZE_EMBEDDINGS, --freeze-embeddings FREEZE_EMBEDDINGS
                        Freeze Embeddings of Model
  -t {multi,answeronly}, --tag {multi,answeronly}
                        Use two different dataset type, multi type and Answer
                        only
  -l2 L2_REGULARIZATION, --l2-regularization L2_REGULARIZATION
                        Value of alpha in l2 regularization 0 means no
                        regularization
  -m {RNNHiddenClassifier,RNNMaxpoolClassifier,RNNFieldClassifier,CNN2dClassifier,CNN1dClassifier,RNNFieldClassifer,CNN1dExtraLayerClassifier}, --model {RNNHiddenClassifier,RNNMaxpoolClassifier,RNNFieldClassifier,CNN2dClassifier,CNN1dClassifier,RNNFieldClassifer,CNN1dExtraLayerClassifier}
                        select the classifier to train on
  -lhd LINEAR_HIDDEN_DIM, --linear-hidden-dim LINEAR_HIDDEN_DIM
                        Freeze Embeddings of Model
```

### Fill In The Blank Generation

Detailed commands can be found in Generation_of_Blanks.ipynb notebook

### Sequence Classification

```zsh
cd FITBGenerator/SequenceClassification
```
#### PreProcess Data
```zsh
python preprocessdata.py
```
Options:
```
usage: preprocessdata.py [-h] [-l LOCATION]

Utility to preprocess the dataset

optional arguments:
  -h, --help            show this help message and exit
  -l LOCATION, --location LOCATION
                        Location of Dataset if left empty configuration will
                        be used
```
#### Train
```zsh
python train.py --help
```
Options: 
```
usage: train.py [-h] [-s SEED] [-loc MODEL_LOCATION] [-b BIDIRECTIONAL]
                [-d DROPOUT] [-e EMBEDDING_DIM] [-hd HIDDEN_DIM] [-l N_LAYERS]
                [-lr LEARNING_RATE] [-n EPOCHS] [-batch BATCH_SIZE]
                [-f FREEZE_EMBEDDINGS] [-l2 L2_REGULARIZATION]
                [-m {RNNHiddenClassifier}] [-lhd LINEAR_HIDDEN_DIM]

Utility to train the Model

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  Set custom seed for reproducibility
  -loc MODEL_LOCATION, --model-location MODEL_LOCATION
                        Give an already trained model location to use and
                        train more epochs on it
  -b BIDIRECTIONAL, --bidirectional BIDIRECTIONAL
                        Makes the model Bidirectional
  -d DROPOUT, --dropout DROPOUT
                        Dropout count for the model
  -e EMBEDDING_DIM, --embedding-dim EMBEDDING_DIM
                        Embedding Dimensions
  -hd HIDDEN_DIM, --hidden-dim HIDDEN_DIM
                        Hidden dimensions of the RNN
  -l N_LAYERS, --n-layers N_LAYERS
                        Number of layers in RNN
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate of Adam Optimizer
  -n EPOCHS, --epochs EPOCHS
                        Number of Epochs to train model
  -batch BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of Epochs to train model
  -f FREEZE_EMBEDDINGS, --freeze-embeddings FREEZE_EMBEDDINGS
                        Freeze Embeddings of Model
  -l2 L2_REGULARIZATION, --l2-regularization L2_REGULARIZATION
                        Value of alpha in l2 regularization 0 means no
                        regularization
  -m {RNNHiddenClassifier}, --model {RNNHiddenClassifier}
                        select the classifier to train on
  -lhd LINEAR_HIDDEN_DIM, --linear-hidden-dim LINEAR_HIDDEN_DIM
                        Freeze Embeddings of Model
```

### Sequence 2 Sequence Generation

```zsh
cd FITBGenerator/Sequence2Sequence
```

#### Preprocessing
```zsh
python preprocess.py --help
```
Options:
```
usage: preprocess.py [-h] [-l LOCATION]

Utility to preprocess the dataset

optional arguments:
  -h, --help            show this help message and exit
  -l LOCATION, --location LOCATION
                        Location of Dataset if left empty configuration will
                        be used
```
#### Train
```zsh
python train.py --help
```
Options:
```
usage: train.py [-h] [-m {LSTM,CNN}] [-n NUM_EPOCHS] [-b BATCH_SIZE]

Utility to Train the model

optional arguments:
  -h, --help            show this help message and exit
  -m {LSTM,CNN}, --model {LSTM,CNN}
                        Select the Seq2Seq Model to train
  -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Number of epochs to train
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Training Batch Size
```




