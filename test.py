import unittest
import dataset
import utils
import app
import pandas as pd


class TestDatasetMethods(unittest.TestCase):
    def test_remove_symptoms(self):
        # First case
        symptoms_1 = "joint pain vomiting yellowish skin dark urine nausea loss of appetite abdominal pain diarrhoea mild fever yellowing of eyes muscle pain".split()
        init_len_1 = len(symptoms_1)
        nb_remove_1 = 4
        dataset.remove_symptoms(symptoms_1, nb_remove_1)
        self.assertEqual(len(symptoms_1), init_len_1 - nb_remove_1)
        # Second case
        symptoms_2 = "continuous sneezing shivering chills watering from eyes".split()
        init_len_2 = len(symptoms_2)
        nb_remove_2 = 7
        dataset.remove_symptoms(symptoms_2, nb_remove_2)
        self.assertEqual(len(symptoms_2), init_len_2 - nb_remove_2)

    def test_is_a_valid_symptom(self):
        # First case
        symptom_1 = "chest pain"
        self.assertTrue(dataset.is_a_valid_symptom(symptom_1))
        # Second case
        symptom_2 = "897538"
        self.assertFalse(dataset.is_a_valid_symptom(symptom_2))
        # Third case
        symptom_3 = "symptoms:46"
        self.assertFalse(dataset.is_a_valid_symptom(symptom_3))
        # Fourth case
        symptom_4 = "9866 chest pain"
        self.assertFalse(dataset.is_a_valid_symptom(symptom_4))
        # Fifth case
        symptom_5 = "chest pain 8647"
        self.assertFalse(dataset.is_a_valid_symptom(symptom_5))
        # Sixth case
        symptom_6 = "346Eye deviation"
        self.assertFalse(dataset.is_a_valid_symptom(symptom_6))

    def test_delete_symptom_id(self):
        # First test
        symptoms_1 = [
            "symptoms:Mass on eyelid",
            "symptoms:84",
            "symptoms:64",
            "symptoms:Chest tightness",
        ]
        symptoms_1_res_expected = [
            "symptoms:Mass on eyelid",
            "symptoms:Chest tightness",
        ]
        dataset.delete_symptoms_id(symptoms_1)
        self.assertEqual(len(symptoms_1), len(symptoms_1_res_expected))
        self.assertEqual(symptoms_1, symptoms_1_res_expected)
        # Second test
        symptoms_2 = [
            "symptoms:Mass on eyelid",
            "symptoms:Chest tightness",
            "symptoms:Palpitations",
        ]
        symptoms_2_res_expected = [
            "symptoms:Mass on eyelid",
            "symptoms:Chest tightness",
            "symptoms:Palpitations",
        ]
        dataset.delete_symptoms_id(symptoms_2)
        self.assertEqual(len(symptoms_2), len(symptoms_2_res_expected))
        self.assertEqual(symptoms_2, symptoms_2_res_expected)
        # Third test
        symptoms_3 = [
            "symptoms:0835 Chest tightness",
            "symptoms:270",
            "symptoms:097",
        ]
        symptoms_3_res_expected = []
        dataset.delete_symptoms_id(symptoms_3)
        self.assertEqual(len(symptoms_3), len(symptoms_3_res_expected))
        self.assertEqual(symptoms_3, symptoms_3_res_expected)

    def test_remove_punctation(self):
        # First test
        medications_1 = dataset.remove_ponctuation(
            "Immun Glob G(IGG)/Pro/Iga 0-50; Sitagliptin; Phosphate"
        )
        medications_1_res_expected = (
            "Immun Glob G(IGG) Pro Iga 0-50 Sitagliptin Phosphate"
        )
        self.assertTrue(medications_1.find("/") == -1)
        self.assertTrue(medications_1.find(";") == -1)
        self.assertEqual(medications_1, medications_1_res_expected)
        # Second test
        medications_2 = dataset.remove_ponctuation(
            "Esomeprazole Magnesium, Trastuzumab, Rivaroxaban, Certolizumab Pegol, Abiraterone Acetate"
        )
        medications_2_res_expected = "Esomeprazole Magnesium, Trastuzumab, Rivaroxaban, Certolizumab Pegol, Abiraterone Acetate"
        self.assertTrue(medications_2.find("/") == -1)
        self.assertTrue(medications_2.find(";") == -1)
        self.assertEqual(medications_2, medications_2_res_expected)
        # Third test
        medications_3 = dataset.remove_ponctuation(";;//;;;;;;;/;/////;;;/")
        medications_3_res_expected = "         "
        self.assertTrue(medications_3.find("/") == -1)
        self.assertTrue(medications_3.find(";") == -1)
        self.assertEqual(medications_3, medications_3_res_expected)

    def test_create_medication_list(self):
        df_medication = pd.read_csv(
            "./datasets/original_datasets/medication.csv", usecols=["generic_name"]
        )
        all_medications = df_medication["generic_name"].values.flatten().tolist()
        # First test
        nb_medications_1 = 9
        medications_1 = dataset.create_medication_list(
            nb_medications_1, all_medications
        ).split(",", nb_medications_1 - 1)
        self.assertEqual(len(medications_1), nb_medications_1)
        # Second test
        nb_medications_2 = 0
        medications_2 = dataset.create_medication_list(
            nb_medications_2, all_medications
        ).split(",")
        self.assertEqual(len(medications_2), nb_medications_2 + 1)


class TestUtilsMethods(unittest.TestCase):
    def test_list_to_string(self):
        # First test
        list_1 = ["I", "have", "a", "dog"]
        string_1_res_expected = "I have a dog"
        string_1 = utils.list_to_string(list_1)
        self.assertEqual(string_1, string_1_res_expected)
        # Second test
        list_2 = []
        string_2_res_expected = ""
        string_2 = utils.list_to_string(list_2)
        self.assertEqual(string_2, string_2_res_expected)
        # Third test
        list_3 = ["89", "1830", "4", "7303086"]
        string_3_res_expected = "89 1830 4 7303086"
        string_3 = utils.list_to_string(list_3)
        self.assertEqual(string_3, string_3_res_expected)


class TestApp(unittest.TestCase):
    def test_generate_prompt(self):
        BASIC_PROMPT = "return me each symptom you see in the following sentence and separate them with a blank space and in lower case. If you don't see a symptom return me 'failed' in lower case:"
        # First test
        input_1 = "My name is Tristan"
        res_1 = app.generate_prompt(input_1)
        res_expected_1 = BASIC_PROMPT + " My name is Tristan"
        self.assertEqual(res_1, res_expected_1)
        # Second test
        input_2 = ""
        res_2 = app.generate_prompt(input_2)
        res_expected_2 = BASIC_PROMPT + " "
        self.assertEqual(res_2, res_expected_2)

    def test_find_medication(self):
        # First test
        disease_1 = "Open wound of the mouth"
        res_1 = app.find_medications(disease_1)
        res_expected_1 = "Enzalutamide, Pemetrexed Disodium, Enalapril Maleate, Sirolimus, Paliperidone Palmitate"
        self.assertEqual(res_1, res_expected_1)
        # Second test
        disease_2 = "No medication test"
        res_2 = app.find_medications(disease_2)
        res_expected_2 = "There is no medications data for this disease"
        self.assertEqual(res_2, res_expected_2)


if __name__ == "__main__":
    unittest.main()
