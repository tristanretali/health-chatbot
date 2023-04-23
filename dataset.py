import pandas as pd
import csv
import os

# Path for each dataset
DISEASE_SYMPTOMS_1 = "./datasets/original_datasets/disease_symptoms.csv"
DISEASE_SYMPTOMS_2 = "./datasets/original_datasets/disease_and_symptoms.csv"
DISEASE_SYMPTOMS_3= "./datasets/original_datasets/symptoms_research_paper.csv"
FORMATTING_DATASET = "./datasets/formatting_dataset.csv"
DATASET = "./datasets/dataset.csv"

def delete_files():
    os.remove(FORMATTING_DATASET)
   # os.remove(DATASET)

def create_dataset():
    """
    Create the final dataset
    """
    disease_symptoms_formatting()
    disease_and_symptoms_formatting()

def disease_symptoms_formatting():
    df = pd.read_csv(DISEASE_SYMPTOMS_1)
    df.drop_duplicates(subset="Disease", keep="first", inplace=True) 
    with open(FORMATTING_DATASET, 'x') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        filewriter.writerow(df.columns)
        for index, row in df.iterrows():
            filewriter.writerow(row)

def disease_and_symptoms_formatting():
    usecols = ["name", "symptoms"]
    df = pd.read_csv(DISEASE_SYMPTOMS_2, usecols=usecols)
    with open(FORMATTING_DATASET, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')       
        for index, row in df.iterrows():
            symptoms = parse_symptoms(row)
            filewriter.writerow(symptoms)

def symptoms_research_paper_formatting():
    usecols=["Symptom", "Disease"]
    all_diseases = {}
    df = pd.read_csv(DISEASE_SYMPTOMS_3, usecols=usecols, delimiter='	')
    with open(FORMATTING_DATASET, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        for index, row in df.iterrows():        
            fill_dict(all_diseases, row)
        
        for value in all_diseases.values():
            filewriter.writerow(value)

def parse_symptoms(row: any) ->list :
    """
    Parse the current row

    Args:
        row (any): the current row of the dataset

    Returns:
        list: return all symptoms of the current row
    """
    symptoms = row["symptoms"].split("{")
    for i in range (len(symptoms)):
        symptoms[i] = symptoms[i][symptoms[i].find(':')+1:symptoms[i].find('}')]
    symptoms[0] = row["name"]
    delete_symptoms_id(symptoms)
    return symptoms

def delete_symptoms_id(symptoms: list):
    """
    Delete the symptoms id of the list

    Args:
        symptoms (list): our current symptoms list
    """
    for i in symptoms:
        if not is_a_valid_symptom(i):
            symptoms.remove(i)

def is_a_valid_symptom(symptom: str) ->bool:
    """
    Check the value of a symptom

    Args:
        symptom (str): the current symptom

    Returns:
        bool: return true iff the symptom is not a number
    """
    for i in symptom:
        if i >= '0' and i <= '9':
            return False
    return True

def fill_dict(all_diseases: dict, row: any):
    if not row["Disease"] in all_diseases :
        all_diseases[row["Disease"]] = [row["Disease"], row["Symptom"]]
    else:
        all_diseases.get(row["Disease"]).append(row["Symptom"]) 

def main():
    delete_files()
    create_dataset()
    symptoms_research_paper_formatting()

if __name__ == "__main__":
    main()