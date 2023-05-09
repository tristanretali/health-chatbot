import pandas as pd
import csv
import os
import random

# Path for each dataset
DISEASE_SYMPTOMS_1 = "./datasets/original_datasets/diseases_symptoms_1.csv"
DISEASE_SYMPTOMS_2 = "./datasets/original_datasets/diseases_symptoms_2.csv"
DISEASE_SYMPTOMS_3 = "./datasets/original_datasets/diseases_symptoms_3.csv"
MEDICATION_DATASET = "./datasets/original_datasets/medication.csv"
FORMATTING_DATASET = "./datasets/formatting_dataset.csv"
DATASET = "./datasets/dataset.csv"
DISEASE_MEDICATION_DATASET = "./datasets/disease_medication_dataset.csv"
FINAL_DATASET = "./datasets/final_dataset.csv"
TEST_DATASET = "./datasets/test_dataset.csv"


def remove_datasets():
    # os.remove(DATASET)
    os.remove(FINAL_DATASET)
    # os.remove(TEST_DATASET)
    # os.remove(DISEASE_MEDICATION_DATASET)


def create_dataset():
    """
    Create the final dataset which include
    symptoms for each diseases
    """
    disease_symptoms_1_formatting()
    disease_symptoms_2_formatting()
    disease_symptoms_3_formatting()


def create_test_dataset():
    data = {}
    df = pd.read_csv(DATASET)
    nb_elements = int(df.shape[0] * 0.10)
    with open(TEST_DATASET, "x") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        filewriter.writerow(df.columns)
        for i in range(nb_elements):
            current_row = df.iloc[random.randint(0, df.shape[0])]
            data[current_row["disease"]] = fill_test_dataset(current_row)
        filewriter.writerows(data.items())


def disease_symptoms_1_formatting():
    """
    Fill the dataset with the disease
    and symptoms data from the first
    dataset
    """
    data = {}
    df = pd.read_csv(DISEASE_SYMPTOMS_1)
    df.drop_duplicates(subset="Disease", keep="first", inplace=True)
    with open(DATASET, "x") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", lineterminator="\n")
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
    with open(DATASET, "a") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", lineterminator="\n")
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
    usecols = ["Symptom", "Disease"]
    data = {}
    df = pd.read_csv(DISEASE_SYMPTOMS_3, usecols=usecols, delimiter="	")
    with open(DATASET, "a") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        for index, row in df.iterrows():
            if not row["Disease"] in data:
                data[row["Disease"]] = f"{row['Symptom']} "
            else:
                data[row["Disease"]] += f"{row['Symptom']} "
        filewriter.writerows(data.items())


def create_final_dataset():
    final_data = []
    df = pd.read_csv(DATASET)
    for index, row in df.iterrows():
        # current_data = [row["disease"], row["symptoms"]]
        all_symptoms = row["symptoms"].split()
        if len(all_symptoms) > 4:
            for i in range(250):
                remove_symptoms_number = random.randint(0, len(all_symptoms) - 2)
                remove_symptoms(all_symptoms, remove_symptoms_number)
                final_data.append([row["disease"], list_to_string(all_symptoms)])
                all_symptoms = row["symptoms"].split()
    fill_final_dataset(final_data)


def remove_symptoms(all_symptoms: list, remove_symptoms_number: int):
    for i in range(remove_symptoms_number):
        all_symptoms.pop(random.randint(0, len(all_symptoms) - 1))


def list_to_string(all_symptoms: list) -> str:
    symptoms = " ".join(all_symptoms)
    return symptoms


def fill_final_dataset(final_data: list):
    with open(FINAL_DATASET, "x") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        filewriter.writerow(["disease", "symptoms"])
        for i in final_data:
            filewriter.writerow(i)


def parse_symptoms(row: any) -> list:
    """
    Parse the current row

    Args:
        row (Series): the current row of the dataset

    Returns:
        list: return all symptoms of the current row
    """
    symptoms = row["symptoms"].split("{")
    for i in range(len(symptoms)):
        symptoms[i] = symptoms[i][symptoms[i].find(":") + 1 : symptoms[i].find("}")]
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


def is_a_valid_symptom(symptom: str) -> bool:
    """
    Check the value of a symptom

    Args:
        symptom (str): the current symptom

    Returns:
        bool: return true iff the symptom is not a number
    """
    for i in symptom:
        if i >= "0" and i <= "9":
            return False
    return True


def is_string_type(symptom) -> bool:
    """
    Check if the symptom is a string

    Args:
        symptom : the current symptom

    Returns:
        bool: return true iff the symptom is a string
    """
    return not isinstance(symptom[1], float)


def fill_test_dataset(current_row) -> str:
    all_symptoms = current_row["symptoms"].split()
    remove_number = random.randint(0, len(all_symptoms) - 1)
    for i in range(remove_number):
        all_symptoms.pop(random.randint(0, len(all_symptoms) - 1))
    symptoms = " ".join(all_symptoms)
    return symptoms


def create_disease_medication_dataset():
    data = {}
    # Open each csv
    df_medication = pd.read_csv(MEDICATION_DATASET, usecols=["generic_name"])
    df_disease = pd.read_csv(FINAL_DATASET, usecols=["disease"])
    # Drop duplicates name
    df_medication.drop_duplicates(subset="generic_name", keep="first", inplace=True)
    df_disease.drop_duplicates(subset="disease", keep="first", inplace=True)
    all_medications = df_medication["generic_name"].values.flatten().tolist()
    for index, row in df_disease.iterrows():
        nb_medication = random.randint(2, 6)
        medications = create_medication_list(nb_medication, all_medications)
        data[row["disease"]] = medications
    fill_disease_medication_dataset(data)


def create_medication_list(nb_medication: int, all_medications: list) -> str:
    medications = ""
    for i in range(nb_medication):
        medication = all_medications[random.randint(0, len(all_medications) - 1)]
        if medications.find(medication) == -1:
            medications += f"{medication} "
    medications = remove_ponctuation(medications)
    return medications


def remove_ponctuation(medications: str) -> str:
    res = medications.replace(";", "").replace("/", " ")
    return res


def fill_disease_medication_dataset(data: dict):
    with open(DISEASE_MEDICATION_DATASET, "x") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",", lineterminator="\n")
        filewriter.writerow(["disease", "medication"])
        filewriter.writerows(data.items())


def main():
    remove_datasets()
    # create_dataset()
    create_final_dataset()
    # create_test_dataset()
    # create_disease_medication_dataset()


if __name__ == "__main__":
    main()
