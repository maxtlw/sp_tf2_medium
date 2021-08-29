import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sptokenizer_layer import SPTokenizer

"""
Train a RNN for text classification as in https://www.tensorflow.org/text/tutorials/text_classification_rnn, but using
the custom SPTokenizer layer for tokenization. The tokenizer is trained on the dataset, using an in-memory iterator.
"""

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Load dataset
dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Instantiate the tokenizer (no model, we want to train it from scratch!)
tokenizer = SPTokenizer()


# To train, we need an iterator which yields strings
def dataset_iterable():
    for x, _ in train_dataset.as_numpy_iterator():
        yield x.decode('utf-8')


tokenizer.train_from_iterable(dataset_iterable(), vocab_size=8000)

# Setup the datasets for training
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build the model: same as the tutorial but with a SP layer instead
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
    tokenizer,
    tf.keras.layers.Embedding(
        input_dim=tokenizer.vocab_size,
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=2,
                    validation_data=test_dataset,
                    validation_steps=30,
                    )

test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss:.2f}')
print(f'Test Accuracy: {test_acc:.2f}')

# Save the model
model.save('mdl')

# TEST: make sure the model loads correctly and can infer
del model
model = tf.keras.models.load_model('mdl')

# Example
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(sample_text)
print(predictions)  # > 0 => positive, < 0 => negative