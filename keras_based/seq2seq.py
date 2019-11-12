"""
Model to introduce errors into previously clean sentences
"""

from keras.layers import Input, LSTM, Dense
from keras.models import Model


def build_model(parameters):
    # encoder
    encoder_inputs = Input(shape=(None, parameters['max_len_in']))
    encoder = LSTM(parameters['encoding_dim'], return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = Input(shape=(None, parameters['max_len_out']))
    decoder_lstm = LSTM(parameters['decoding_dim'], return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(parameters['max_len_out'], activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model
