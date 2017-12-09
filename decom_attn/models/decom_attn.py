import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
from keras.layers.merge import concatenate
import keras.backend as K

from config import (
    BiMPMConfig, TrainConfig
)
from char_repres import CharRepresLayer


MAX_LEN = TrainConfig.MAX_SEQUENCE_LENGTH
#MAX_LEN = 40


def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    #pretrained_weights = np.load(open(pretrained_weights_path,'rb'))
    pretrained_weights = pretrained_weights_path
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_= Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
                             output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_embedding, word_index, char_index = None,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='relu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933
    
    nb_per_word = TrainConfig.MAX_CHAR_PER_WORD
    rnn_unit = BiMPMConfig.RNN_UNIT
    #nb_words = min(TrainConfig.MAX_NB_WORDS, len(word_index)) + 1
    #word_embedding_dim = TrainConfig.WORD_EMBEDDING_DIM
    dropout = BiMPMConfig.DROP_RATE

    if TrainConfig.USE_CHAR:
        nb_chars = min(TrainConfig.MAX_NB_CHARS, len(char_index)) + 1
        char_embedding_dim = TrainConfig.CHAR_EMBEDDING_DIM
        char_rnn_dim = TrainConfig.CHAR_LSTM_DIM

    q1 = Input(name='q1',shape=(maxlen,))
    q2 = Input(name='q2',shape=(maxlen,))

    if TrainConfig.USE_CHAR:
        c1 = Input(shape=(maxlen, nb_per_word), dtype='int32')
        c2 = Input(shape=(maxlen, nb_per_word), dtype='int32')
    
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, 
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)
    #c1_embed = embedding(c1)
    #c2_embed = embedding(c2)

    # Model chars input
    if TrainConfig.USE_CHAR:
        char_layer = CharRepresLayer(
            maxlen, nb_chars, nb_per_word, char_embedding_dim,
            char_rnn_dim, rnn_unit=rnn_unit, dropout=dropout)
        c_res1 = char_layer(c1)
        c_res2 = char_layer(c2)
        sequence1 = concatenate([q1_embed, c_res1])
        sequence2 = concatenate([q2_embed, c_res2])
    else:
        sequence1 = q1_embed
        sequence2 = q2_embed
    
    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
                Dense(projection_hidden, activation=activation),
                Dropout(rate=projection_dropout),
            ])
    projection_layers.extend([
            Dense(projection_dim, activation=None),
            Dropout(rate=projection_dropout),
        ])
    q1_encoded = time_distributed(sequence1, projection_layers)
    q2_encoded = time_distributed(sequence2, projection_layers)
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)    
    
    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)]) 
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)
    
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    if TrainConfig.USE_CHAR:
        inputs = (q1, q2, c1, c2)
    else:
        inputs = (q1, q2)
    
    model = Model(inputs=inputs, outputs=out_)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy'])
    print(model.summary())
    return model
