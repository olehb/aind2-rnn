import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
from string import ascii_lowercase


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    rows = series.size-window_size
    X = np.zeros((rows, window_size))
    y = np.zeros((rows, 1))

    for i in range(rows):
        X[i] = series[i:i+window_size]
        y[i] = series[i+window_size]

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(4, input_shape=(window_size,1)))
    model.add(Dense(1))
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    text = text.lower()

    cleaned_text = np.zeros(len(text), dtype=np.uint8)
    cleaned_text_length = 0
    
    allowed_characters = set(list(ascii_lowercase + ' ') + punctuation)
    space_code = ord(' ')

    for i in range(len(text)):
        cleaned_text[i] = ord(text[i]) if text[i] in allowed_characters else space_code

    return cleaned_text.tobytes().decode('utf8')

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text)-window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
