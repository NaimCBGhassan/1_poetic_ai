import random
import numpy as np
import tensorflow as tf
# Reasignaciones personalizadas
Sequential = tf.keras.models.Sequential
LSTM, Dense, Activation = tf.keras.layers.LSTM, tf.keras.layers.Dense, tf.keras.layers.Activation
RMSprop = tf.keras.optimizers.RMSprop

# Descargar el texto
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]


# Procesamiento de texto
charachters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(charachters)}
index_to_char = {i: c for i, c in enumerate(charachters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

# sentences = []
# next_characters = []

# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i: i + SEQ_LENGTH])
#     next_characters.append(text[i + SEQ_LENGTH])

# x = np.zeros((len(sentences), SEQ_LENGTH, len(charachters)), dtype=bool)
# y = np.zeros((len(sentences), len(charachters)), dtype=bool)

# for i, sentence in enumerate(sentences):
#     for t, character in enumerate(sentence):
#         x[i, t, char_to_index[character]] = 1
#     y[i, char_to_index[next_characters[i]]] = 1

# # Convertir los datos a float32
# x = x.astype('float32')
# y = y.astype('float32')

# # Construcción del modelo
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(charachters))))
# model.add(Dense(len(charachters), activation = 'relu'))
# model.add(Activation('softmax'))

# # Compilación del modelo
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# # Entrenamiento del modelo
# model.fit(x, y, batch_size=256, epochs=10)

# # Guardar el modelo
# model.save('poetic.model.v2')

# load model
model = tf.keras.models.load_model('poetic.model')

# Generar texto
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, diversity):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence

    for i in range(length):
        x_pred = np.zeros((1, SEQ_LENGTH, len(charachters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# ---0.3---
print("---0.3---")
print(generate_text(300, 0.3))
# ---0.6---
print("---0.6---")
print(generate_text(300, 0.6))
# ---1.0---
print("---1.0---")
print(generate_text(300, 1.0))