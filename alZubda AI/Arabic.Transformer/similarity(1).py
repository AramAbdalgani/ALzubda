import scipy.spatial.distance
import numpy as np
from sentence_transformers import SentenceTransformer
import arabic_reshaper
import bidi
from bidi.algorithm import get_display

#model = SentenceTransformer('distiluse-base-multilingual-cased')
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

corpus = [
    "أنا أحب التعلم والتطوير",
    "اللغة العربية عبارة عن لغة رائعة",
   "أريد أن أصبح مهندسا في المستقبل", "أتذكر الأيام الجميلة في المدرسة",
]
corpus_embeddings = model.encode(corpus)

queries = [
    "ادرس اللغة العربية في الجامعة",
]
query_embeddings = model.encode(queries)

# Calculate Cosine similarity of query against each sentence in corpus
num_results = 3
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = sorted(enumerate(distances), key=lambda x: x[1])

    print("\n======================\n")
    reshaped_text = arabic_reshaper.reshape(query) # reshaping text
    bidi_text_OUT = get_display(reshaped_text)
    print("Query:", bidi_text_OUT)
    print("\nTop", num_results, "most similar sentences in corpus:")

    for idx, distance in results[:num_results]:
        reshaped_text = arabic_reshaper.reshape(corpus[idx].strip()) # reshaping text
        bidi_text = get_display(reshaped_text)
        print(bidi_text, "(Score: {:.4f})".format(1-distance))