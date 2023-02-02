import csv
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from textaugment import Wordnet
from textaugment import Translate
from textaugment import EDA
import random

file_path = "C:/Users/Admin/Desktop/alZubda AI/data augmentation/Book2.csv"
v = False 
n = True 
runs = 5  # increased the number of runs
p = 0.5 
src = "en" 
to = "fr" 

with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['Text']
        print("Original text: ", text)
        
        for i in range(runs):
            t = Wordnet(v=v ,n=n, p=p)
            augmented_text = t.augment(text)
            print("Wordnet augmented text: ", augmented_text)

            t = Translate(src=src, to=to)
            augmented_text = t.augment(text)
            print("Translate augmented text: ", augmented_text)

            t = EDA()
            augmented_text = t.synonym_replacement(text)
            print("EDA Synonym Replacement augmented text: ", augmented_text)

            t = EDA()
            augmented_text = t.random_deletion(text, p=0.2)
            print("EDA Random Deletion augmented text: ", augmented_text)

            t = EDA()
            augmented_text = t.random_swap(text)
            print("EDA Random Swap augmented text: ", augmented_text)

            t = EDA()
            augmented_text = t.random_insertion(text)
            print("EDA Random Insertion augmented text: ", augmented_text)