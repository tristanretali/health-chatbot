import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D, SpatialDropout1D, LSTM

SEQUENCES_PADDING = 11  #  180
VOCAB_SIZE = 740
EMBEDDING_DIM = 25
EPOCH = 5
BATCH_SIZE = 64

df_train = pd.read_csv("./datasets/final_dataset.csv").iloc[:1000]


def clean_symptoms_data(symptoms: str) -> str:
    # Lower case for each symptom
    symptoms = symptoms.lower()

    # Remove work containing less than 3 characters
    my_symptoms = symptoms.split()

    # Loop on a copy to avoid the index problem
    for symptom in my_symptoms[:]:
        if len(symptom) <= 3:
            my_symptoms.remove(symptom)
    symptoms = list_to_string(my_symptoms)

    # Remove pre and post Blank Spaces
    symptoms = symptoms.strip()

    return symptoms


def list_to_string(my_symptoms: list) -> str:
    symptoms = ""
    for symptom in my_symptoms:
        symptoms += f"{symptom} "
    return symptoms


df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)

# Change each sentences into a sequence of integer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df_train["symptoms"].values)
word_index = tokenizer.word_index
# print(f"Found {len(word_index)} unique tokens.")
X = tokenizer.texts_to_sequences(df_train["symptoms"].values)
X = pad_sequences(X, maxlen=SEQUENCES_PADDING)
# print(f"Shape of data tensor: {X.shape}")

# Convert each label into number
Y = pd.get_dummies(df_train["disease"]).values
# print(f"Shape of label tensor: {Y.shape}")
print(df_train["disease"][0], Y[0])
print(df_train["disease"][250], Y[250])
print(df_train["disease"][500], Y[500])
print(df_train["disease"][750], Y[750])
# Split data for the training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42
)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)


# model = Sequential()
# # Use 50 length vectors to represent each work
# model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQUENCES_PADDING))
# model.add(SpatialDropout1D(0.2))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(4, activation="softmax"))
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.summary()
# history = model.fit(
#     X_train,
#     Y_train,
#     epochs=EPOCH,
#     batch_size=BATCH_SIZE,
#     validation_split=0.1,
#     callbacks=[EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001)],
# )

# accr = model.evaluate(X_test, Y_test)
# print("Test loss:", accr[0])
# print("Test accuracy:", accr[1])

# plt.title("Loss")
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="test")
# plt.legend()
# plt.show()
# plt.title("Accuracy")
# plt.plot(history.history["accuracy"], label="train")
# plt.plot(history.history["val_accuracy"], label="test")
# plt.legend()
# plt.show()

# # Each label used for the model training
# labels = ["Allergy", "Chronic cholestasis", "Fungal infection", "GERD"]
# # Test model prediction
# sentences1 = ["itching skin rash nodal skin eruptions dischromic patches"]
# sentences2 = ["stomach pain acidity ulcers on tongue vomiting cough chest pain"]
# seq1 = tokenizer.texts_to_sequences(sentences1)
# seq2 = tokenizer.texts_to_sequences(sentences2)
# padded1 = pad_sequences(seq1, maxlen=SEQUENCES_PADDING)
# print(f"padded1: {padded1}")
# padded2 = pad_sequences(seq2, maxlen=SEQUENCES_PADDING)
# print(f"padded2: {padded2}")
# pred1 = model.predict(padded1)
# print(labels)
# print(pred1, labels[np.argmax(pred1)])
# pred2 = model.predict(padded2)
# print(pred1, labels[np.argmax(pred2)])


def set_label(df_train, Y):
    df_train["disease"].iloc[:1000].drop_duplicates().values.flatten().tolist()
    # np.vstack(list(set(tuple(row) for row in Y)))  np.unique(Y, axis=0)
    labels_number = []
    for i in Y:
        if list(i) not in labels_number:
            labels_number.append(list(i))
    print(labels_number)


set_label(df_train, Y)
