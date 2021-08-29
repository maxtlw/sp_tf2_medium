import pathlib

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow_text import SentencepieceTokenizer, Tokenizer, pad_model_inputs

CURRENT_PATH = pathlib.Path(__file__).parent
MODEL_PATH = CURRENT_PATH.joinpath('sp_alice.model')
model = gfile.GFile(MODEL_PATH, 'rb').read()

tf_sp = SentencepieceTokenizer(
    model=model,
    #alpha=0.1,
    #nbest_size=-1,
    #add_bos=True,
    #add_eos=True,
    #reverse=False
)


def tokenize(text_batch: tf.Tensor,
             tokenizer: Tokenizer,
             make_lower: bool = True,
             max_sequence_length: int = 512,
             fixed_length: bool = False
             ) -> tf.Tensor:
    # Possibly make lowercase
    if make_lower:
        text_batch = tf.strings.lower(text_batch)
    # Tokenize
    tokenized_batch = tokenizer.tokenize(text_batch)
    # If we need to pad/truncate to max length
    if fixed_length:
        seq_length = max_sequence_length
    else:
        tokenized_batch_max_length = tf.cast(tokenized_batch.bounding_shape(axis=1), dtype=tf.int32)
        seq_length = tf.minimum(max_sequence_length, tokenized_batch_max_length)

    tokenized_batch, _ = pad_model_inputs(tokenized_batch,
                                          max_seq_length=seq_length,
                                          pad_value=0
                                          )

    return tokenized_batch


input_ = tf.constant(
    ["'Yes, that's it,' said the Hatter with a sigh: 'it's always tea-time'",
     "But if I'm not the same, the next question is, Who in the world am I?"]
)

tokenized_input = tokenize(input_, tf_sp)
print(tokenized_input)


