import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from string import ascii_lowercase


def window_transform_series(series, window_size):
    rows = series.size-window_size
    X = np.zeros((rows, window_size))
    y = np.zeros((rows, 1))

    for i in range(rows):
        X[i] = series[i:i+window_size]
        y[i] = series[i+window_size]

    return X, y


def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(4, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = text.lower()

    cleaned_text = np.zeros(len(text), dtype=np.uint8)

    allowed_characters = set(list(ascii_lowercase + ' ') + punctuation)
    space_code = ord(' ')

    for i in range(len(text)):
        cleaned_text[i] = ord(text[i]) if text[i] in allowed_characters else space_code

    return cleaned_text.tobytes().decode('utf8')


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text)-window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs, outputs


def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
