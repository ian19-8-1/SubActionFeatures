import torch
from transformers import BertTokenizer, BertModel
import numpy

import Utils


print("Loading sub-action descriptions...")
txt_file_path = "sub-actions.txt"
sub_actions = Utils.get_data(txt_file_path)


print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


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
for (i, sa) in enumerate(sa_embeds):
    sa_embeds[i] = sa.detach().numpy()
sa_embeds = numpy.asarray(sa_embeds)

numpy.savetxt('embeddings.csv', sa_embeds, delimiter=',')


print('Finished')
