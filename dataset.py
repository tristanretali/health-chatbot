import pandas as pd
import numpy as np
import csv
import os

DISEASE_SYMPTOMES = "./datasets/disease_symptomes.csv"
DISEASE_SYMPTOMES_FORMATTING = "./datasets/disease_symptomes_formatting.csv"

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
            print(index)
    #     if not row[0] in diseases_informations.keys():
    #         diseases_informations[row[0]] = [] 
    #     for values in row.items():
    #         #if row[i] != "":
    #         if not values[1] in diseases_informations.get(row[0]):
    #             diseases_informations.get(row[0]).append(values[1])
    # print(diseases_informations.get("Fungal infection"))    
                #print(values)
           #print('Disease is: {}; Symptoms : {}'.format(row["Disease"], row["Symptom_1"]))
        #filewriter.writerow(['Derek', 'Software Developer'])
        #filewriter.writerow(['Steve', 'Software Developer'])
        #filewriter.writerow(['Paul', 'Manager'])
    #print(df.columns)



def main():
    delete_files()
    disease_symptomes_formatting()

if __name__ == "__main__":

    main()