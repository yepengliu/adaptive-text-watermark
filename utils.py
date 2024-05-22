import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model.eval()
    return model, tokenizer

def pre_process(dataset, min_length, data_size=500):
    data = []
    for text in dataset['train']['text']:
        text0 = text.split()[0:min_length]
        if len(text0) >= min_length:
            text0 = ' '.join(text0)
            data.append({'text0': text0, 'text': text})
        else:
            pass
        
        if len(data) ==  data_size:
            break

    return data