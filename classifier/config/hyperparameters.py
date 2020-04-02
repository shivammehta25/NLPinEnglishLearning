"""
Hyper Parameters of Model
"""

MAX_VOCAB = 10000
BATCH_SIZE = 64
# Keep it 300 since we are using glove 300d vectors
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 2
BIDIRECTION = True
DROPOUT = 0.7
LR = 0.001
EPOCHS = 5