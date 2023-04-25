import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET = "./datasets/dataset.csv"

df_train = pd.read_csv(DATASET)

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

print("Before data treatment : " + df_train["symptoms"][71] + "\n")

df_train["symptoms"] = df_train["symptoms"].apply(clean_symptoms_data)

print("After data treatment : " + df_train["symptoms"][71])

df_train['word_count'] = df_train['symptoms'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10,8))
sns.histplot(data=df_train, x='word_count')
plt.title('Word Count of Symptoms in Train Data after data treatment')
plt.xlabel('Word Count')
plt.ylabel('Symptom Count')
plt.show()
