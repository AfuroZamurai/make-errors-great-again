"""
Train a model
"""

from seq2seq import build_model
from preprocessing import get_charwise_encoder_input, get_charwise_decoder_input, get_charwise_decoder_output

BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2


if __name__ == '__main__':
    encoder_input_data = get_charwise_encoder_input()
    decoder_input_data = get_charwise_decoder_input()
    decoder_target_data = get_charwise_decoder_target()

    params = {
        'max_len_in': 100,
        'max_len_out': 100,
        'encoding_dim': 300,
        'decoding_dim': 300,
        'vocab_size': 100
    }
    model = build_model(params)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit()
