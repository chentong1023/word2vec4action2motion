from transformers import BertTokenizer, BertModel
import torch
from scipy.spatial.distance import cosine

def word_embedding(sequence, tokenizer, model):
	token = tokenizer(sequence, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**token)
		hidden_states = outputs[2]
	token_embeddings = torch.stack(hidden_states, dim=0)
	token_embeddings = torch.squeeze(token_embeddings, dim=1)
	token_embeddings = token_embeddings.permute(1,0,2)
#	print(token_embeddings.size())
	token_vecs_sum = []
	for token in token_embeddings:
		sum_vec = torch.sum(token[-4:], dim=0)
		token_vecs_sum.append(sum_vec)
	return token_vecs_sum

def sequence_embedding(sequence, tokenizer, model):
	token = tokenizer(sequence, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**token)
		hidden_states = outputs[2]
	token_vecs = hidden_states[-2][0]
	sentence_embed = torch.mean(token_vecs, dim=0)
	return sentence_embed