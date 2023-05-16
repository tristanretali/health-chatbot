import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import wandb
from sklearn.model_selection import train_test_split

# from sklearn import preprocessing
# import tensorflow as tf
import os

# import visualkeras
# from tensorflow import keras
from keras.preprocessing.text import Tokenizer

# from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM, Dropout
import mlflow
import mlflow.keras

# from keras.layers import GlobalAveragePooling1D, SpatialDropout1D, LSTM
# from wandb.keras import WandbMetricsLogger

SEQUENCES_PADDING = 100  # 180
VOCAB_SIZE = 763
EMBEDDING_DIM = 200
EPOCH = 6
BATCH_SIZE = 256  # 64 16

df_train = pd.read_csv("./datasets/final_dataset.csv").iloc[:250000]


"""wandb.init(
    project="health-chatbot",
    entity="tristan-retali",
    config={
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": EPOCH,
        "batch_size": BATCH_SIZE,
    },
)
config = wandb.config"""


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


# df_train["nb_word"] = df_train["symptoms"].apply(lambda x: len(x.split()))
# show_word_distribution(df_train)

# Start login
mlflow.start_run()

df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)

# Change each sentences into a sequence of integer
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(df_train["symptoms"].values)
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")
X = tokenizer.texts_to_sequences(df_train["symptoms"].values)
X = pad_sequences(X, maxlen=SEQUENCES_PADDING)
print(f"Shape of data tensor: {X.shape}")

# Convert each label into number
Y = pd.get_dummies(df_train["disease"]).values
# print(f"Shape of label tensor: {Y.shape}")
# print(df_train["disease"][0], Y[0])
# print(df_train["disease"][250], Y[250])
# print(df_train["disease"][500], Y[500])
# print(df_train["disease"][750], Y[750])
# Split data for the training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
# Use 100 length vectors to represent each work
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQUENCES_PADDING))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(1500, dropout=0.3, recurrent_dropout=0.3))  # 4518
model.add(Dense(1024, activation="relu"))  # 4518
model.add(Dropout(0.3))
model.add(Dense(977, activation="softmax"))  # 4518
model.summary()

# visualkeras.layered_view(model, legend=True, to_file="neural_network.png")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Add WandbMetricsLogger to log metrics and WandbModelCheckpoint to log model checkpoints

history = model.fit(
    X_train,
    Y_train,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001)],
)


# model.save("model")
score = model.evaluate(X_test, Y_test)

mlflow.log_metric("Final test loss", score[0])
print(f"Test loss: {score[0]}")

mlflow.log_metric(f"Final test accuracy", score[1])
print(f"Test accuracy {score[1]}")

mlflow.log_metric("Epoch number", EPOCH)
mlflow.log_metric("Batch size number", BATCH_SIZE)


fig = plt.figure(figsize=(6, 3))
plt.title("Health Chatbot with Keras ({} epochs)".format(EPOCH), fontsize=14)
plt.plot(history.history["val_accuracy"], "b-", label="Accuracy", lw=4, alpha=0.5)
plt.plot(history.history["val_loss"], "r--", label="Loss", lw=4, alpha=0.5)
plt.legend(fontsize=14)
plt.grid(True)

# Log an image
mlflow.log_figure(fig, "accuracy_and_loss.png")

# Registering the model to the workspace
print("Registering the model via MLFlow")
registered_model_name = "health-chatbot-977-10-epochs"

mlflow.keras.log_model(
    model=model,
    registered_model_name=registered_model_name,
    artifact_path=registered_model_name,
    extra_pip_requirements=["protobuf~=3.20"],
)

# saving the model to a file
print("Saving the model via MLFlow")
mlflow.keras.save_model(
    model=model,
    path=os.path.join(registered_model_name, "trained_model"),
    extra_pip_requirements=["protobuf~=3.20"],
)

mlflow.end_run()


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


def set_label(df_train, Y):
    labels = list(
        range(len(df_train["disease"].drop_duplicates().values.flatten().tolist()))
    )
    for i in range(0, Y.shape[0], 250):
        labels[list(Y[i]).index(1)] = df_train["disease"][i]
    return labels


# model_register = keras.models.load_model("model")

# Each label used for the model training
labels = set_label(df_train, Y)
# Test model prediction
for i in range(0, len(df_train["symptoms"]), 250):
    sentences = [df_train["symptoms"][i]]
    seq = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(seq, maxlen=SEQUENCES_PADDING)
    print(f"padded: {padded}")
    pred = model.predict(padded)
    print(f"Should be {df_train['disease'][i]}: {labels[np.argmax(pred)]}\n {pred}")
# sentences1 = [
#     "The sickness person complains of fatigue and muscle pain all over their body, as well as a hoarse voice and regurgitation. They also experience neck and mouth pain, along with symptoms of a groin mass and an ulcer. Additionally, they have a lump mass and itchy ear."
# ]
# sentences2 = ["I feel a stomach pain an I have ulcer on tong"]
# seq1 = tokenizer.texts_to_sequences(sentences1)
# seq2 = tokenizer.texts_to_sequences(sentences2)
# padded1 = pad_sequences(seq1, maxlen=SEQUENCES_PADDING)
# print(f"test: {labels[np.argmax(pred)]}")
# padded2 = pad_sequences(seq2, maxlen=SEQUENCES_PADDING)
# print(f"padded2: {padded2}")
# pred1 = model.predict(padded1)
# print(labels)
# print(pred1, labels[np.argmax(pred1)])
# pred2 = model.predict(padded2)
# print(pred2, labels[np.argmax(pred2)])
