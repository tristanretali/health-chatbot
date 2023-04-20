import pandas as pd
import csv
import os

DISEASE_SYMPTOMS_1 = "./datasets/original_datasets/disease_symptoms.csv"
DISEASE_SYMPTOMS_2 = "./datasets/original_datasets/disease_and_symptoms.csv"
PRECAUTIONS = "./datasets/original_datasets/symptom_precaution.csv"
DISEASE_SYMPTOMS_FORMATTING = "./datasets/formatting_datasets/disease_symptoms_formatting.csv"

def delete_files():
    os.remove(DISEASE_SYMPTOMS_FORMATTING)

def disease_symptoms_formatting():
    df = pd.read_csv(DISEASE_SYMPTOMS_1)
    df.drop_duplicates(subset="Disease", keep="first", inplace=True) 
    with open(DISEASE_SYMPTOMS_FORMATTING, 'x') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        filewriter.writerow(df.columns)
        for index, row in df.iterrows():
            filewriter.writerow(row)

def disease_and_symptoms_formatting():
    usecols = ["name", "symptoms"]
    df = pd.read_csv(DISEASE_SYMPTOMS_2, usecols=usecols)
    with open(DISEASE_SYMPTOMS_FORMATTING, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')       
        for index, row in df.iterrows():
            symptoms = parse_symptoms(row)
            filewriter.writerow(symptoms)
            
            
def parse_symptoms(row: any) ->list :
    symptoms = row["symptoms"].split("{")
    for i in range (len(symptoms)):
        symptoms[i] = symptoms[i][symptoms[i].find(':')+1:symptoms[i].find('}')]
    symptoms[0] = row["name"]
    delete_symptoms_id(symptoms)
    return symptoms

def delete_symptoms_id(symptoms: list):
    for i in symptoms:
        if not is_a_valid_symptom(i):
            symptoms.remove(i)


def is_a_valid_symptom(symptom: str) ->bool:
    for i in symptom:
        if i >= '0' and i <= '9':
            return False
    return True


def main():
    delete_files()
    disease_symptoms_formatting()
    disease_and_symptoms_formatting()

if __name__ == "__main__":
    main()