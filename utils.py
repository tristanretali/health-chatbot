import pandas as pd
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

df_symptoms = pd.read_csv("./datasets/final_dataset.csv").iloc[:250000]

# Common constant
SEQUENCES_PADDING = 100
VOCAB_SIZE = 740
Y = pd.get_dummies(df_symptoms["disease"]).values


def list_to_string(all_symptoms: list) -> str:
    """
    Convert a list of symptoms in one string

    Args:
        all_symptoms (list): the symptoms who will be in the string

    Returns:
        str: return the string with all symptoms
    """
    symptoms = " ".join(all_symptoms)
    return symptoms


def create_tokenizer() -> Tokenizer:
    """
    Generate a tokenizer object based on my vocabulary size

    Returns:
        Tokenizer: return the tokenizer associated
    """
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(df_symptoms["symptoms"].values)
    return tokenizer
