import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATASET = "./datasets/dataset.csv"
SEQUENCES_PADDING = 180


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
    print(f"This is the percentage of symptoms information for a disease who have less than {train_symptoms}%")

def x_train_treatment(X_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_train_sequences = pad_sequences(X_train_sequences, padding="post", maxlen=SEQUENCES_PADDING)
    #print(X_train[0])
    #print(X_train_sequences[0])
    #print(len(X_train_sequences[0]))
    #print(len(X_train_sequences[1]))
    return X_train_sequences

def main():
    df_train = pd.read_csv(DATASET)
    #print("Before data treatment : " + df_train["symptoms"][71] + "\n")
    df_train["nb_word"] = df_train['symptoms'].apply(lambda x: len(x.split()))
    df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)
    #print("After data treatment : " + df_train["symptoms"][71])
    #show_word_distribution(df_train)
    #percentage_word_count(df_train, 180)
    X_train = df_train["symptoms"]
    Y_train = df_train["disease"]
    x_train_sequences = x_train_treatment(X_train)

if __name__ == "__main__":
    main()
