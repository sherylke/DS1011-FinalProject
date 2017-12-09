from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
#from keras.layers.wrappers import Bidirectional
#from keras.legacy.layers import Highway
#from keras.layers import TimeDistributed
import keras.backend as K
#from keras.layers.normalization import BatchNormalization

class CharRepresLayer(object):
    """Char embedding representation layer
    """
    def __init__(self, sequence_length, nb_chars, nb_per_word,
                 embedding_dim, rnn_dim, rnn_unit='gru', dropout=0.0):
        def _collapse_input(x, nb_per_word=0):
            x = K.reshape(x, (-1, nb_per_word))
            return x

        def _unroll_input(x, sequence_length=0, rnn_dim=0):
            x = K.reshape(x, (-1, sequence_length, rnn_dim))
            return x

        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(Lambda(_collapse_input,
                              arguments={'nb_per_word': nb_per_word},
                              output_shape=(nb_per_word,),
                              input_shape=(sequence_length, nb_per_word,)))
        self.model.add(Embedding(nb_chars,
                                 embedding_dim,
                                 input_length=nb_per_word,
                                 trainable=True))
        self.model.add(rnn(rnn_dim,
                           dropout=dropout,
                           recurrent_dropout=dropout))
        self.model.add(Lambda(_unroll_input,
                              arguments={'sequence_length': sequence_length,
                                         'rnn_dim': rnn_dim},
                              output_shape=(sequence_length, rnn_dim)))
        
    def __call__(self, inputs):
        return self.model(inputs)