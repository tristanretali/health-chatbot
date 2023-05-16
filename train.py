import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM, Dropout
import mlflow
import mlflow.keras


SEQUENCES_PADDING = 100
VOCAB_SIZE = 763
EMBEDDING_DIM = 200
EPOCH = 6
BATCH_SIZE = 256

df_train = utils.df_symptoms


def clean_symptoms_data(symptoms: str) -> str:
    """
    Standardize each symptoms string

    Args:
        symptoms (str): the current symptoms

    Returns:
        str: return the input normalize
    """
    # Lower case for each symptom
    symptoms = symptoms.lower()
    # Remove work containing less than 3 characters
    my_symptoms = symptoms.split()
    # Loop on a copy to avoid the index problem
    for symptom in my_symptoms[:]:
        if len(symptom) <= 3:
            my_symptoms.remove(symptom)
    symptoms = utils.list_to_string(my_symptoms)
    # Remove pre and post Blank Spaces
    symptoms = symptoms.strip()
    return symptoms


# Apply the clean to each row of my dataset
df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)

# Start login
mlflow.start_run()

# Change each sentences into a sequence of integer
tokenizer = utils.create_tokenizer()
word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")
X = tokenizer.texts_to_sequences(df_train["symptoms"].values)
X = pad_sequences(X, maxlen=SEQUENCES_PADDING)
print(f"Shape of data tensor: {X.shape}")

# Apply the one hot encoder to my labels(diseases)
# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, utils.Y, test_size=0.10, random_state=42
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
# Use 200 length vectors to represent each work
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQUENCES_PADDING))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(1500, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(977, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    X_train,
    Y_train,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001)],
)


score = model.evaluate(X_test, Y_test)

# Put in the metric area of Azure ML Studio the final test loss
mlflow.log_metric("Final test loss", score[0])
print(f"Test loss: {score[0]}")

# Put in the metric area of Azure ML Studio the final test accuracy
mlflow.log_metric(f"Final test accuracy", score[1])
print(f"Test accuracy {score[1]}")

# Put in the metric area of Azure ML Studio my epoch number and my batch size
mlflow.log_metric("Epoch number", EPOCH)
mlflow.log_metric("Batch size number", BATCH_SIZE)


# Create a graph who will be add in th Azure ML studio
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

# Finish the mlflow run
mlflow.end_run()
