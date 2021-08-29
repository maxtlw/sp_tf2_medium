import io
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import sentencepiece as spm
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow_text import SentencepieceTokenizer, pad_model_inputs


class SPTokenizer(tf.keras.layers.Layer):
    """ SentencePiece tokenizer layer. """
    def __init__(self,
                 model_file_path: Optional[Union[str, Path]] = None,
                 make_lower: bool = True,
                 max_sequence_length: int = 512,
                 fixed_length: bool = False,
                 training_alpha: float = 1.0,
                 training_nbest_size: int = 1,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 reverse: bool = False,
                 use_tf_function: bool = True   # Switch off eager mode for better performance
                 ):
        super(SPTokenizer, self).__init__()
        self.make_lower = make_lower
        self.max_sequence_length = max_sequence_length
        self.fixed_length = fixed_length
        self._training_alpha = training_alpha
        self._training_nbest_size = training_nbest_size
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._reverse = reverse
        self.use_tf_function = use_tf_function

        self._tokenizer = None
        self._tokenizer_is_set = False
        if model_file_path is not None:
            self.load_from_file(model_file_path)

    def _set_tokenizer(self, model: bytes):
        if self._tokenizer_is_set:
            warnings.warn('Tokenizer was already set.')
        tokenizer = SentencepieceTokenizer(model,
                                           alpha=self._training_alpha,
                                           add_bos=self._add_bos,
                                           add_eos=self._add_eos,
                                           reverse=self._reverse
                                           )
        self._tokenizer = tokenizer
        self._tokenizer_is_set = True

    def load_from_file(self, model_file_path):
        model = gfile.GFile(model_file_path, 'rb').read()
        self._set_tokenizer(model)

    def train_from_iterable(self, input_iterable: Iterable, vocab_size: int = 8000):
        model = io.BytesIO()
        spm.SentencePieceTrainer.Train(sentence_iterator=input_iterable,
                                       model_writer=model,
                                       vocab_size=vocab_size,
                                       pad_id=0,
                                       unk_id=1,
                                       bos_id=2,
                                       eos_id=3
                                       )
        self._set_tokenizer(model.getvalue())

    def train_from_file(self, input_file_path: str, vocab_size: int = 8000):
        model = io.BytesIO()
        spm.SentencePieceTrainer.Train(input=input_file_path,
                                       model_writer=model,
                                       vocab_size=vocab_size,
                                       pad_id=0,
                                       unk_id=1,
                                       bos_id=2,
                                       eos_id=3
                                       )
        self._set_tokenizer(model.getvalue())

    def get_config(self):
        return {
            **super(SPTokenizer, self).get_config(),
            'make_lower': self.make_lower,
            'max_seqence_length': self.max_sequence_length,
            'fixed_length': self.fixed_length,
            'training_alpha': self._training_alpha,
            'training_nbest_size': self._training_nbest_size,
            'add_bos': self._add_bos,
            'add_eos': self._add_eos,
            'reverse': self._reverse,
            'use_tf_function': self.use_tf_function
        }

    @property
    def vocab_size(self):
        if not self._tokenizer_is_set:
            raise Exception(
                'Tokenizer model not found. Please load a SentencePiece model file or train the tokenizer on a corpus.')
        return int(self._tokenizer.vocab_size())

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def _tf_tokenize(self, text_batch: tf.Tensor) -> tf.Tensor:
        return self._tokenize(text_batch)

    def _tokenize(self, text_batch: tf.Tensor) -> tf.Tensor:
        if not self._tokenizer_is_set:
            raise Exception(
                'Tokenizer model not found. Please load a SentencePiece model file or train the tokenizer on a corpus.')

        # Possibly make lowercase
        if self.make_lower:
            text_batch = tf.strings.lower(text_batch)
        # Tokenize
        tokenized_batch = self._tokenizer.tokenize(text_batch)
        # If we need to pad/truncate to max length
        if self.fixed_length:
            seq_length = self.max_sequence_length
        else:
            tokenized_batch_max_length = tf.cast(tokenized_batch.bounding_shape(axis=1), dtype=tf.int32)
            seq_length = tf.minimum(self.max_sequence_length, tokenized_batch_max_length)

        tokenized_batch, _ = pad_model_inputs(tokenized_batch,
                                              max_seq_length=seq_length,
                                              pad_value=0
                                              )

        return tokenized_batch

    def call(self, inputs, training: bool = True, **kwargs):
        if training:
            self._tokenizer.nbest_size = self._training_nbest_size
        else:
            self._tokenizer.nbest_size = 1

        if self.use_tf_function:
            return self._tf_tokenize(inputs)
        return self._tokenize(inputs)
