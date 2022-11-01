from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Bidirectional



def build_model_lstm(input_dim, output_dim, return_sequences):
    """
    Builds an improved Long Short term memory model using keras.layers.recurrent.lstm
    :param input_dim: input dimension of model
    :param output_dim: ouput dimension of model
    :param return_sequences: return sequence for the model
    :return: a 3 layered LSTM model
    """
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.1))

    model.add(LSTM(
        100,
        return_sequences=False))

    model.add(Dropout(0.1))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model




def build_model_gru(input_dim, output_dim, return_sequences):
    """
    Builds a basic lstm model 
    :param input_dim: input dimension of the model
    :param output_dim: output dimension of the model
    :param return_sequences: return sequence of the model
    :return: a basic lstm model with 3 layers.
    """
    model = Sequential()
    model.add(GRU(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.1))

    model.add(GRU(
        100,
        return_sequences=False))

    model.add(Dropout(0.1))
   
    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model



def build_model_lgru(input_dim, output_dim, return_sequences):
    """
    Builds a LGRU model 
    :param input_dim: input dimension of the model
    :param output_dim: output dimension of the model
    :param return_sequences: return sequence of the model
    :return: a basic lstm model with 3 layers.
    """
    model = Sequential()


    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.1))


    model.add(GRU(
        100,
        return_sequences=False))

    model.add(Dropout(0.1))


    model.add(Dense(
        units=1))
    model.add(Activation('linear'))


    return model


def build_model_bilgru(input_dim, output_dim, return_sequences):
    
    model = Sequential()


    model = Sequential()
    model.add(Bidirectional(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences)))

    model.add(Dropout(0.1))


    model.add(GRU(
        100,
        return_sequences=False))

    model.add(Dropout(0.1))


    model.add(Dense(
        units=1))
    model.add(Activation('linear'))


    return model
    
    
def build_model_bilbigru(input_dim, output_dim, return_sequences):
    
    model = Sequential()


    model = Sequential()
    model.add(Dense(512, activation='linear'))
    model.add(Dense(256, activation='linear'))
    model.add(Dense(128, activation='linear'))
    model.add(Dense(64, activation='linear'))
    model.add(Dense(32, activation='linear'))

    model.add(Bidirectional(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences)))

    model.add(Dropout(0.2))


    model.add(Bidirectional(GRU(
        100,
        return_sequences=False)))

    model.add(Dropout(0.2))


    model.add(Dense(
        units=1))
    model.add(Activation('linear'))


    return model
    
def build_model_bilbigrumlp(input_dim, output_dim, return_sequences):
    
    model = Sequential()

    model = Sequential()
    model.add(Bidirectional(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences)))

    model.add(Dropout(0.2))
    model.add(Dense(64, activation='linear')) 
    model.add(Dense(128, activation='linear'))
    model.add(Dense(256, activation='linear'))
    model.add(Dense(512, activation='linear'))

    model.add(Bidirectional(GRU(
        100,
        return_sequences=False)))

    model.add(Dropout(0.2))
    model.add(Dense(32, activation='linear')) 
    model.add(Dense(64, activation='linear'))
    model.add(Dense(
        units=1))
    model.add(Activation('linear'))


    return model