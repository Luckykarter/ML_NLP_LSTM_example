from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle

"""
This is an example of using single layer LSTM, dual layer LSTM
and convolutional model on the prepared dataset from 
tensorflow_datasets
"""

BUFFER_SIZE = 10000
BATCH_SIZE = 64
TRAIN_EPOCHS = 10
MODEL_TYPE = 'single_lstm'
# MODEL_TYPE = 'dual_lstm'
# MODEL_TYPE = 'conv'


def plot_graphs(history, name):
    plt.plot(history[name])
    plt.plot(history['val_' + name])
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.legend([name, 'val_' + name])
    plt.show()


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

tokenizer = info.features['text'].encoder

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(
    BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))

test_dataset = test_dataset.padded_batch(
    BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))

model_path = MODEL_TYPE + '_model.h5'
train_history_path = MODEL_TYPE + '_train_history'

plot_history = None
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    if os.path.exists(train_history_path):
        plot_history = pickle.load(open(train_history_path, 'rb'))
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, 64)])

    if MODEL_TYPE == 'single_lstm':
        model.add(tf.keras.layers.LSTM(64))
    elif MODEL_TYPE == 'conv':
        model.add(tf.keras.layers.Conv1D(128, 5, activation=tf.nn.relu))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
    elif MODEL_TYPE == 'dual_lstm':
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)))
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)))

    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=TRAIN_EPOCHS, validation_data=test_dataset)
    plot_history = history.history
    with open(train_history_path, 'wb') as file:
        pickle.dump(plot_history, file)
    model.save(model_path)

if plot_history:
    plot_graphs(plot_history, 'accuracy')
    plot_graphs(plot_history, 'loss')
