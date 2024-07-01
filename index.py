                        # tokenisation and data preparation


#  Tokenisation -> This involves converting the Hindi words 
#  into sequences of characters that the model can process.

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

# Load the data
data = pd.read_csv('.dataset.csv')
# store correct and incorrect in list format
misspelled_words = data['misspelled'].tolist()
correct_words = data['correct'].tolist()

# Create tokenizers
input_tokenizer = Tokenizer(char_level=True)
output_tokenizer = Tokenizer(char_level=True)

# Fit tokenizers on data(o/p will be a long list of characters ['a','b'])
input_tokenizer.fit_on_texts(misspelled_words)
output_tokenizer.fit_on_texts(correct_words)

# Convert words to sequences(each character in list is converted into a number sequnce [[1,2,3],[4,5,6]])
input_sequences = input_tokenizer.texts_to_sequences(misspelled_words)
output_sequences = output_tokenizer.texts_to_sequences(correct_words)

# Get vocabulary sizes
# word index is used to map each char with the seqeunce like {'ा': 1, 'र': 2,} to form a dictionary 
input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

# Find maximum sequence lengths
max_input_length = max(len(seq) for seq in input_sequences)
max_output_length = max(len(seq) for seq in output_sequences)

# Pad sequences
input_data = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
output_data = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

# Create one-hot encoded output
output_data_one_hot = np.zeros((len(output_data), max_output_length, output_vocab_size), dtype='float32')

for i, sequence in enumerate(output_data):
    for t, char_index in enumerate(sequence):
        if char_index > 0:
            output_data_one_hot[i, t, char_index] = 1.

# Set random seed for reproducibility
tf.random.set_seed(42)
embedding_dim = 256
num_heads = 8
ff_dim = 512
num_transformer_blocks = 4
dropout_rate = 0.1

def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(inputs, inputs)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embedding_dim)(ffn_output)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_model(input_vocab_size, output_vocab_size, max_input_length, max_output_length):
    inputs = layers.Input(shape=(max_input_length,))
    embedding_layer = layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dim)
    x = embedding_layer(inputs)
    
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, num_heads, ff_dim, dropout_rate)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_vocab_size * max_output_length, activation="softmax")(x)
    outputs = layers.Reshape((max_output_length, output_vocab_size))(outputs)
    
    return models.Model(inputs=inputs, outputs=outputs)

# Build the model
model = build_model(input_vocab_size, output_vocab_size, max_input_length, max_output_length)

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)

# Train the model
history = model.fit(
    input_data,
    output_data_one_hot,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Load the best model
best_model = models.load_model('best_model.keras')

def correct_spelling(word):
    input_seq = input_tokenizer.texts_to_sequences([word])
    input_seq = pad_sequences(input_seq, maxlen=max_input_length, padding='post')
    predicted = best_model.predict(input_seq)
    predicted_seq = np.argmax(predicted[0], axis=-1)
    corrected_word = output_tokenizer.sequences_to_texts([predicted_seq])[0]
    return corrected_word.strip()

# Test the model
test_words = ["accomodate"]
for word in test_words:
    print(f"Original: {word}, Corrected: {correct_spelling(word)}")