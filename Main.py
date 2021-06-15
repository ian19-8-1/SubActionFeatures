import torch
from transformers import BertTokenizer, BertModel
import pickle

import Utils


print("Loading sub-action descriptions...")
txt_file_path = "sub-actions.txt"
sub_actions = Utils.get_data(txt_file_path)


print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# for sa in sub_actions:
#     print(tokenizer.tokenize(sa))
# exit()

print('Formatting inputs...')
# Utils.get_max_len(sub_actions, tokenizer)
max_len = 7
input_ids, attn_masks = Utils.convert_inputs(sub_actions, tokenizer, max_len)


print('Loading BERT model...')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


print('Embedding tokens...')
model.eval()
outputs = model(input_ids=input_ids,
                attention_mask=attn_masks)
hidden_states = outputs[2]

sa_embeds = []
for sa in hidden_states[-2]:
    sa_embeds.append(torch.mean(sa, dim=0))


print('Saving embeddings...')
dict = {}
for i in range(len(sa_embeds)):
    dict[i+1] = sa_embeds[i]
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(dict, f)


print('Finished')
