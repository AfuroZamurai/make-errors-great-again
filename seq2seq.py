"""
Model to introduce errors into previously clean sentences
"""

from keras.layers import Input, LSTM, Dense
from keras.models import Model


def build_model(parameter):
    # encoder
    encoder_inputs = Input(shape=(None, parameter['max_len_in']))
    encoder = LSTM(parameter['encoding_dim'], return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = Input(shape=(None, parameter['max_len_out']))
    decoder_lstm = LSTM(parameter['decoding_dim'], return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(parameter['vocab_size'], activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # model
    model = Model([encoder_inputs, encoder_outputs], decoder_outputs)

    return model


if __name__ == '__main__':
    p = {
        'max_len_in': 100,
        'max_len_out': 120,
        'encoding_dim': 300,
        'decoding_dim': 300,
        'vocab_size': 100
    }
    model = build_model(p)
    model.summary()
