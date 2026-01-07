import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, AveragePooling1D,
    LSTM, Dense, Dropout, Reshape
)
from tensorflow.keras.regularizers import l2
from training.config import L2_REG, DROPOUT_RATE


def build_cnn_lstm(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(128, 2, activation='relu', padding='same',
               kernel_regularizer=l2(L2_REG))(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, padding='same')(x)

    x = Conv1D(128, 2, activation='relu', padding='same',
               kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, padding='same')(x)

    x = Conv1D(512, 2, activation='relu', padding='same',
               kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, padding='same')(x)

    x = Conv1D(512, 2, activation='relu', padding='same',
               kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, padding='same')(x)

    x = Reshape((6, 256))(x)

    x = LSTM(256, return_sequences=True,
             kernel_regularizer=l2(L2_REG))(x)
    x = LSTM(256, return_sequences=True,
             kernel_regularizer=l2(L2_REG))(x)
    x = LSTM(256, kernel_regularizer=l2(L2_REG))(x)

    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(L2_REG))(x)

    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
