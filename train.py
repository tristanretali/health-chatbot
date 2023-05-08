import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D, LSTM, SpatialDropout1D, GlobalAveragePooling1D
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

DATASET = "./datasets/dataset.csv"
TEST_DATASET = "./datasets/test_dataset.csv"
SEQUENCES_PADDING = 180
VOCAB_SIZE = 739
OUTPUT_DIM = 50
EPOCH = 20
BATCH_SIZE = 64


# authenticate
credential = DefaultAzureCredential()
# # Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="75c73b60-6890-4ba6-9a83-83a1589ae4e5",
    resource_group_name="tristan.retali-rg",
    workspace_name="health-chatbot-models",
)

def clean_symptoms_data(symptoms: str) -> str:
    
     #Lower case for each symptom
    symptoms = symptoms.lower()

    # Remove work containing less than 3 characters
    my_symptoms = symptoms.split()

    # Loop on a copy to avoid the index problem
    for symptom in my_symptoms[:]:
        if len(symptom) <= 3:
            my_symptoms.remove(symptom)
    symptoms = list_to_string(my_symptoms)

    #Remove pre and post Blank Spaces
    symptoms = symptoms.strip()

    return symptoms

def list_to_string(my_symptoms: list) ->str:
    symptoms = ""
    for symptom in my_symptoms:
        symptoms += f"{symptom} "
    return symptoms

def show_word_distribution(df_train):
    plt.figure(figsize=(10,8))
    sns.histplot(data=df_train, x='nb_word')
    plt.title('Word Count of Symptoms in Train Data after data treatment')
    plt.xlabel('Word Count')
    plt.ylabel('Symptom Count')
    plt.show()

def percentage_word_count(df_train, nb_word: int):
    train_symptoms = (sum(df_train["nb_word"] < nb_word)/df_train.shape[0])*100
    print(f"This is the percentage of symptoms information for a disease who have less than {nb_word} words: {train_symptoms}%")

def x_treatment(x_value):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(x_value)
    x_train_sequences = tokenizer.texts_to_sequences(x_value)
    x_train_sequences = pad_sequences(x_train_sequences, padding="post", maxlen=SEQUENCES_PADDING)

    #print(f"The vocabulary size: {len(tokenizer.index_word) + 1}")
    #print(x_value[0])
    #print(x_train_sequences[0])
    return x_train_sequences

'''def y_treatment(y_value):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(y_value)
    y_train_sequences = tokenizer.texts_to_sequences(y_value)
    return y_train_sequences '''


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, input_length=SEQUENCES_PADDING, output_dim=OUTPUT_DIM))
    '''model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])'''
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    #output_array = model.predict(x_value)
    #print(output_array)
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.1)
    model.save("model")
    score = model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

def main():
    df_train = pd.read_csv(DATASET)
    '''df_test = pd.read_csv(TEST_DATASET)
    #print("Before data treatment : " + df_train["symptoms"][71] + "\n")
    df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)
    df_test["symptoms"] = df_test["symptoms"].apply(clean_symptoms_data)
    #print("After data treatment : " + df_train["symptoms"][71])
    #df_train["nb_word"] = df_train['symptoms'].apply(lambda x: len(x.split()))
    #show_word_distribution(df_train)
    #percentage_word_count(df_train, 180)
    #Data to train the model
    x_train_sequences = x_treatment(df_train["symptoms"].iloc[:10])
    #print(df_train["symptoms"].iloc[:10])
    #print(x_train_sequences)
    y_train = pd.get_dummies(df_train["disease"].iloc[:10], dtype=int).values
    #print(y_train)
    #Data to test the model
    x_test_sequences = x_treatment(df_test["symptoms"].iloc[:10])
    y_test = pd.get_dummies(df_test["disease"].iloc[:10], dtype=int).values
    #print(x_test_sequences)
    #print(y_test)
    #print(x_train_sequences.shape, y_train.shape)
    create_model(x_train_sequences, y_train, x_test_sequences, y_test)'''
    model = keras.models.load_model('model')
    tokenizer = Tokenizer()
    test = ["fatigue weight loss restlessness lethargy irregular sugar level blurred and distorted vision obesity excessive hunger increased appetite polyuria"]
    seq = tokenizer.texts_to_sequences(test)
    padded = pad_sequences(seq, maxlen=SEQUENCES_PADDING)
    labels = df_train["disease"].iloc[:10].values.flatten().tolist()
    #print(labels)
    pred = model.predict(padded)
    print(pred, labels[np.argmax(pred)])

if __name__ == "__main__":
    main()
