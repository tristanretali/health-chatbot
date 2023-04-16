import pandas as pd
import numpy as np
import csv
import os

DISEASE_SYMPTOMES = "./datasets/disease_symptomes.csv"
PRECAUTION = "./datasets/symptom_precaution.csv"
DISEASE_SYMPTOMES_FORMATTING = "./datasets/disease_symptomes_formatting.csv"
DATASET = "dataset.csv"

def delete_files():
    os.remove(DISEASE_SYMPTOMES_FORMATTING)

def disease_symptomes_formatting():
    df = pd.read_csv(DISEASE_SYMPTOMES)
    df.drop_duplicates(subset="Disease", keep="first", inplace=True) 
    with open("./datasets/disease_symptomes_formatting.csv", 'x') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(df.columns)
        for index, row in df.iterrows():
            filewriter.writerow(row)




def main():
    delete_files()
    disease_symptomes_formatting()

if __name__ == "__main__":

    main()