import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D, SpatialDropout1D, LSTM


DATASET = "./datasets/final_dataset.csv"
TEST_DATASET = "./datasets/test_dataset.csv"
SEQUENCES_PADDING = 11  #  180
VOCAB_SIZE = 740
OUTPUT_DIM = 50
EPOCH = 5
BATCH_SIZE = 64


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


def show_word_distribution(df_train):
    plt.figure(figsize=(10, 8))
    sns.histplot(data=df_train, x="nb_word")
    plt.title("Word Count of Symptoms in Train Data after data treatment")
    plt.xlabel("Word Count")
    plt.ylabel("Symptom Count")
    plt.show()


def percentage_word_count(df_train, nb_word: int):
    train_symptoms = (sum(df_train["nb_word"] < nb_word) / df_train.shape[0]) * 100
    print(
        f"This is the percentage of symptoms information for a disease who have less than {nb_word} words: {train_symptoms}%"
    )


def x_treatment(x_train):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(x_train)
    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_train_sequences = pad_sequences(
        x_train_sequences, padding="post", maxlen=SEQUENCES_PADDING
    )
    # print(f"The vocabulary size: {len(tokenizer.index_word) + 1}")
    # print(x_train[0])
    # print(x_train_sequences[0])
    return x_train_sequences


"""def y_treatment(y_value):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(y_value)
    y_train_sequences = tokenizer.texts_to_sequences(y_value)
    return y_train_sequences """


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    # input_dim: size of the vocabulary, input_lenght: maximum length of a sequence, output_dim: length of the vector for each word
    model.add(
        Embedding(
            input_dim=VOCAB_SIZE, input_length=x_train.shape[1], output_dim=OUTPUT_DIM
        )
    )

    model.add(Conv1D(128, 5, activation="relu"))
    # GlobalAveragePooling1D: decide if the model should keep the temporal dimension
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(4, activation="softmax"))

    # model.add(Dense(512, input_shape=(SEQUENCES_PADDING,), activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(4, activation="softmax"))

    # model.add(SpatialDropout1D(0.2))
    # # LSTM: Long Short-Term Memory, contain a memory cell and 3 door :
    # # - Input door: decide if the input should change the content of the cell
    # # - Oblivion door: decide if the content of the cell should be 0
    # # - Output door: decide if the content of the cell should influate on the output of the neuron
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dense(64, activation="relu"))
    # model.add(Dense(4, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()
    model.fit(
        x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.1
    )
    # output_array = model.predict(x_train)
    # print(output_array)
    model.save("model")
    score = model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def main():
    df_train = pd.read_csv(DATASET)
    # df_test = pd.read_csv(TEST_DATASET)
    # print("Before data treatment : " + df_train["symptoms"][71] + "\n")
    df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)
    # df_test["symptoms"] = df_test["symptoms"].apply(clean_symptoms_data)
    # print("After data treatment : " + df_train["symptoms"][71])
    # df_train["nb_word"] = df_train['symptoms'].apply(lambda x: len(x.split()))
    # show_word_distribution(df_train)
    # percentage_word_count(df_train, 180)
    # Data to train the model
    x = x_treatment(df_train["symptoms"].iloc[:1000])
    # print(df_train["symptoms"].iloc[:10])
    # print(x_train_sequences)
    y = pd.get_dummies(df_train["disease"].iloc[:1000], dtype=int).values
    # print(x)
    # print(y)
    # text_size: percentage for the test data from the current data, random_state for shuffling
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.10, random_state=42
    )
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    # print(x_train_sequences.shape, y_train.shape)
    create_model(x_train, y_train, x_test, y_test)
    model = keras.models.load_model("model")
    tokenizer = Tokenizer()
    test = ["itching skin rash nodal eruptions dischromic patches"]
    seq = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(seq, maxlen=SEQUENCES_PADDING)
    labels = df_train["disease"].iloc[:1000].drop_duplicates().values.flatten().tolist()
    print(labels)
    pred = model.predict(padded)
    print(pred, labels[np.argmax(pred)])
    test = ["stomach pain acidity ulcers on tongue vomiting cough chest pain"]
    seq = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(seq, maxlen=SEQUENCES_PADDING)
    labels = df_train["disease"].iloc[:1000].drop_duplicates().values.flatten().tolist()
    print(labels)
    pred = model.predict(padded)
    print(pred, labels[np.argmax(pred)])


if __name__ == "__main__":
    main()
