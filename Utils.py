import torch


def get_data(file_path):

    # Get list of sub-action descriptions
    with open(file_path, "r") as f:
        sub_actions = f.readlines()

    # Remove newlines
    for (i, sa) in enumerate(sub_actions):
        sub_actions[i] = sa[:-1]

    return sub_actions


def get_max_len(sub_actions, tokenizer):
    max_len = 0
    for sa in sub_actions:
        ids = tokenizer.encode(sa, add_special_tokens=True)

        max_len = max(max_len, len(ids))

    print("Max length: ", max_len)


def convert_inputs(sub_actions, tokenizer, max_len):
    input_ids = []
    attn_masks = []

    for sa in sub_actions:
        encoded_dict = tokenizer.encode_plus(
                                                sa,
                                                add_special_tokens=True,
                                                max_length=max_len,
                                                truncation=True,
                                                padding='max_length',
                                                return_attention_mask=True,
                                                return_tensors='pt',
                                            )

        input_ids.append(encoded_dict['input_ids'])
        attn_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attn_masks = torch.cat(attn_masks, dim=0)

    return input_ids, attn_masks
