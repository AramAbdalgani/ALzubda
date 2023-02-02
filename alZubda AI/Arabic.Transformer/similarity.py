from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Sentences we want sentence embeddings for
#sentences = ['السلام عليكم', 'مرحبا بكم']
sentences = ['ذهبت للتسوق في الامس', 'أنا أحب الحيوانات']

tokenizer = AutoTokenizer.from_pretrained('dangvantuan/sentence-camembert-large')
model = AutoModel.from_pretrained('dangvantuan/sentence-camembert-large')


encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


with torch.no_grad():
    model_output = model(**encoded_input)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1])
sentence_embeddings[0].unsqueeze(0)
print("Similarity between sentence 0 and sentence 1: ", similarity.item())
print("Tokenized Input Sentences: ", encoded_input['input_ids'])
print("Attention Masks: ", encoded_input['attention_mask'])
#print("Sentence embeddings:")
#print(sentence_embeddings)