import csv
import arabic_reshaper
from bidi.algorithm import get_display
import pyarabic.araby as araby
import csv
import nltk
from textaugment import Translate
from textaugment import Wordnet
from textaugment import Translate
from textaugment import EDA
from nltk import ngrams
import augly.text as textaugs
import magic
nltk.download('punkt')
nltk.download('wordnet')


file_path = "data augmentation/arabic_data.csv"
v = False 
n = True 
runs = 1  # increased the number of runs
p = 0.5 
n_value = 3
src = "ar" 
to = "fr" 

def preprocess_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text) # reshaping text
    bidi_text = get_display(reshaped_text) # handling bidirectional text
    return bidi_text

augmented_texts = set()
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['Text']
        normalized_text = araby.strip_tashkeel(text)
        preprocessed_text = preprocess_arabic_text(normalized_text)
        print("Original text: ", preprocessed_text)
        
        for i in range(runs):
            t = Wordnet(v=v ,n=n, p=p)
            n_grams = ngrams(normalized_text.split(), n_value)
            n_grams_text = " ".join([" ".join(ngram) for ngram in n_grams])
            reshaped_n_grams_text = preprocess_arabic_text(n_grams_text)
            if reshaped_n_grams_text not in augmented_texts:
                print("N-grams text: ",reshaped_n_grams_text)
                augmented_texts.add(reshaped_n_grams_text)

            t = Translate(src=src, to=to)
            augmented_text = t.augment(text)
            reshaped_augmented_text = preprocess_arabic_text(augmented_text)
            if reshaped_augmented_text not in augmented_texts:
                print("Translate augmented text: ", reshaped_augmented_text)
                augmented_texts.add(reshaped_augmented_text)

            t = EDA()
            augmented_text = t.synonym_replacement(text)
            reshaped_augmented_text = preprocess_arabic_text(augmented_text)
            if reshaped_augmented_text not in augmented_texts:
                print("EDA Synonym Replacement augmented text: ", reshaped_augmented_text)
                augmented_texts.add(reshaped_augmented_text)

            t = EDA()
            augmented_text = t.random_deletion(text, p=0.2)
            reshaped_augmented_text = preprocess_arabic_text(augmented_text)
            if reshaped_augmented_text not in augmented_texts:
                print("EDA Random Deletion augmented text: ", reshaped_augmented_text)
                augmented_texts.add(reshaped_augmented_text)

            t = EDA()
            augmented_text = t.random_swap(text)
            reshaped_augmented_text = preprocess_arabic_text(augmented_text)
            if reshaped_augmented_text not in augmented_texts:
                print("EDA Random Swap augmented text: ", reshaped_augmented_text)
                augmented_texts.add(reshaped_augmented_text)

            t = EDA()
            augmented_text = t.random_insertion(text)
            reshaped_augmented_text = preprocess_arabic_text(augmented_text)
            if reshaped_augmented_text not in augmented_texts:
                print("EDA Random Insertion augmented text: ", reshaped_augmented_text)
                augmented_texts.add(reshaped_augmented_text)


            
        

               






            
            