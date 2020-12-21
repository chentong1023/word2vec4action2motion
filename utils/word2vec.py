import os
import torch
from transformers import BertTokenizer, BertModel
from models.mlp import MLP
from embed.word_embeding import sequence_embedding

def init_network():
	model_path = ""
	model_file_path = os.path.join(model_path, "latest.tar")
	model = torch.load(model_file_path)
	project_net = MLP(768, 768, [128, 64])
	project_net.load_state_dict(model["project_net"])
	bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	bert_model = BertModel.from_pretrained(
		"bert-base-uncased", output_hidden_states=True
	)
	bert_model.eval()
	project_net.eval()

def word_embeding_pair(word, bert_tokenizer, bert_model, project_net):
	category_em = sequence_embedding(word, bert_tokenizer, bert_model)
	cate_embed_project = project_net(cate_embed)
	return category_em, cate_embed_project