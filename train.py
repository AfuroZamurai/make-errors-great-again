"""
Train a model
"""
import os

from seq2seq import build_model
from preprocessing import Preprocessor

BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2


if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data', 'smallest.csv')
    preprocessor = Preprocessor(data_path)
    encoder_input_data, decoder_input_data, decoder_target_data = preprocessor.get_charwise_data()

    params = {
        'max_len_in': 93,
        'max_len_out': 78,
        'encoding_dim': 300,
        'decoding_dim': 300,
        'vocab_size': 100
    }
    model = build_model(params)
    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=VALIDATION_SPLIT)
