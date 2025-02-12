"""
Train a model
"""
import os

from keras.models import load_model

from keras_based.seq2seq import build_model
from keras_based.preprocessing import Preprocessor

BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LOAD_MODEL = True


if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data', 'smallest.csv')
    preprocessor = Preprocessor(data_path)
    encoder_input_data, decoder_input_data, decoder_target_data = preprocessor.get_charwise_data()

    params = {
        'max_len_in': 78,
        'max_len_out': 79,
        'encoding_dim': 300,
        'decoding_dim': 300
    }

    if LOAD_MODEL:
        model = load_model('final.h5')
    else:
        model = build_model(params)
        model.summary()

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_split=VALIDATION_SPLIT)
