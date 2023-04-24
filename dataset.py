import pandas as pd
import csv
import os

# Path for each dataset
DISEASE_SYMPTOMS_1 = "./datasets/original_datasets/diseases_symptoms_1.csv"
DISEASE_SYMPTOMS_2 = "./datasets/original_datasets/diseases_symptoms_2.csv"
DISEASE_SYMPTOMS_3= "./datasets/original_datasets/diseases_symptoms_3.csv"
FORMATTING_DATASET = "./datasets/formatting_dataset.csv"
DATASET = "./datasets/dataset.csv"

def create_dataset():
    """
    Create the final dataset which include 
    symptoms for each diseases
    """
    disease_symptoms_1_formatting()
    disease_symptoms_2_formatting()
    disease_symptoms_3_formatting()

def disease_symptoms_1_formatting():
    """
    Fill the dataset with the disease
    and symptoms data from the first 
    dataset
    """
    data = {}
    df = pd.read_csv(DISEASE_SYMPTOMS_1)
    df.drop_duplicates(subset="Disease", keep="first", inplace=True) 
    with open(DATASET, 'x') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        filewriter.writerow(["disease", "symptoms"])
        for index, row in df.iterrows():
            for symptom in row.items():
                if is_string_type(symptom):
                    if not row["Disease"] in data:                        
                        data[row["Disease"]] = ""
                    else:
                        data[row["Disease"]] += f"{symptom[1]} "
        filewriter.writerows(data.items())

def disease_symptoms_2_formatting():
    """
    Fill the dataset with the disease
    and symptoms data from the second 
    dataset
    """
    usecols = ["Disease", "symptoms"]
    data = {}
    df = pd.read_csv(DISEASE_SYMPTOMS_2, usecols=usecols)
    with open(DATASET, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')       
        for index, row in df.iterrows():
            symptoms = parse_symptoms(row)
            data[row["Disease"]] = ""
            for symptom in symptoms:
                data[row["Disease"]] += f"{symptom} "
        filewriter.writerows(data.items())

def disease_symptoms_3_formatting():
    """
    Fill the dataset with the disease
    and symptoms data from the third 
    dataset
    """
    usecols=["Symptom", "Disease"]
    data = {}
    df = pd.read_csv(DISEASE_SYMPTOMS_3, usecols=usecols, delimiter='	')
    with open(DATASET, 'a') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        for index, row in df.iterrows():        
            if not row["Disease"] in data:
                data[row["Disease"]] = f"{row['Symptom']} "
            else:
                data[row["Disease"]] += f"{row['Symptom']} "
        filewriter.writerows(data.items())

def parse_symptoms(row: any) ->list :
    """
    Parse the current row

    Args:
        row (Series): the current row of the dataset

    Returns:
        list: return all symptoms of the current row
    """
    symptoms = row["symptoms"].split("{")
    for i in range (len(symptoms)):
        symptoms[i] = symptoms[i][symptoms[i].find(':')+1:symptoms[i].find('}')]
    del symptoms[0]
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

def is_string_type(symptom) ->bool:
    """
    Check if the symptom is a string
    
    Args:
        symptom : the current symptom

    Returns:
        bool: return true iff the symptom is a string
    """
    return not isinstance(symptom[1], float)

def main():
    os.remove(DATASET)
    create_dataset()

if __name__ == "__main__":
    main()