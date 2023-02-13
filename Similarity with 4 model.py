from flask import Flask, request, jsonify
import scipy.spatial.distance
import numpy as np
from sentence_transformers import SentenceTransformer
import arabic_reshaper
import bidi
from bidi.algorithm import get_display
from scipy.spatial.distance import pdist

app = Flask(__name__)

model1 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model2 = SentenceTransformer('AIDA-UPM/MSTSb_stsb-xlm-r-multilingual')
model3 = SentenceTransformer('mrm8488/distiluse-base-multilingual-cased-v2-finetuned-stsb_multi_mt-es')
model4 = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')


@app.route('/process_input', methods=['POST'])
def process_input():
    # global corpus # declare corpus as global
    
    data = request.get_json()

    queries = data['queries']
    corpus = data['corpus']
    num_results = data.get('num_results', 6)
    
    print("-----------------")
    print(queries)
    print(num_results)
    print(corpus)

    preprocessed_corpus_embeddings = [model.encode(corpus) for model in [model1,model2,model3,model4]]
    preprocessed_query_embeddings = [model.encode(queries) for model in [model1,model2,model3,model4]]

    # The number of results to show
    num_results = 6

    # Compute the cosine similarities between the preprocessed query embeddings and the preprocessed corpus embeddings
    similarities = [1 - scipy.spatial.distance.cdist(preprocessed_corpus_embeddings[i], preprocessed_query_embeddings[i], "cosine").flatten()
                    for i in range(4)]

    # Sort the similarities for each query embeddings
    sorted_similarities = [sorted(enumerate(similarity), key=lambda x: x[1], reverse=True) for similarity in similarities]

    # Compute the final results based on the similarities
    final_results = []
    for i in range(len(corpus)):
        scores = [sorted_similarity[i][1] for sorted_similarity in sorted_similarities]
        count = 0
        for score in scores:
            if score >= 0.65:
                count += 1
        if count >= 2:
            final_results.append((sorted_similarities[0][i][0], max(scores)))
        else:
            final_results.append((sorted_similarities[0][i][0], min(scores)))

    # Sort the final results
    final_results = sorted(final_results, key=lambda x: x[1], reverse=True)[:num_results]


    print("\n======================\n")
    accepted_results=[]
    rejected_results=[]
    print("\nTop", num_results, "most similar sentences in corpus:")
    for (corpus_idx, distance) in final_results[:num_results]:
        reshaped_text = arabic_reshaper.reshape(corpus[corpus_idx].strip())
        bidi_text = (reshaped_text)
        if distance >= 0.55:
            print(bidi_text, " Accepted")
            accepted_results.append(bidi_text)
        else:
            print(bidi_text, " Rejected")
            rejected_results.append(bidi_text)

    
    return jsonify({'Accepted': accepted_results, 'Rejected': rejected_results})


if __name__ == "__main__":
    app.run(debug=True)
    
    
    