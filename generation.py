import torch
from sentence_transformers import SentenceTransformer
from model import SemanticModel
import json
import argparse
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt')

from utils import load_model, pre_process, vocabulary_mapping
from watermark import Watermark



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load semantic embedding model
    embedding_model = SentenceTransformer(args.embedding_model).to(device)
    embedding_model.eval()
    # load semantic mapping model
    transform_model = SemanticModel()
    transform_model.load_state_dict(torch.load(args.semantic_model))
    transform_model.to(device)
    transform_model.eval()
    # load mapping list
    vocalulary_size = 50272  # vacalulary size of LLM
    mapping_list = vocabulary_mapping(vocalulary_size, 384)
    # load test dataset. Here we use C4 realnewslike dataset as an example. Feel free to use your own dataset.
    data = ""
    dataset = load_dataset('json', data_files=data)
    dataset = pre_process(dataset, min_length=args.min_new_tokens, data_size=500)   # [{text0: 'text0', text: 'text'}]



    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embedding_model=embedding_model,
                      transform_model=transform_model,
                      mapping_list=mapping_list,
                      alpha=args.alpha,
                      top_k=50,
                      top_p=0.9,
                      repetition_penalty=1.1,
                      no_repeat_ngram_size=0,
                      max_new_tokens=args.max_new_tokens,
                      min_new_tokens=args.min_new_tokens,
                      secret_string=args.secret_string,
                      measure_threshold=args.measure_threshold,
                      delta_0 = args.delta_0,
                      delta = args.delta,
                      )
    
    df = pd.DataFrame(columns=['text_id', 'prompt', 'human', 'unwatermarked_text', 'adaptive_watermarked_text', 'paraphrased_text',\
                                'human_score', 'adaptive_watermarked_text_score', 'paraphrased_text_score'])

    for i in tqdm(range(len(dataset))):
        human = dataset[i]['text0']
        prompt = ' '.join(nltk.sent_tokenize(dataset[i]['text'])[:2])   # first two sentences
        
        unwatermarked_text = watermark.generate_unwatermarked(prompt)
        watermarked_text = watermark.generate_adaptive_watermarke(prompt)
        p_watermarked_text = watermark.paraphrase(openai_api_key=args.openai_api_key, input_text=watermarked_text)

        human_score = watermark.detection(human)
        adaptive_watermarked_text_score = watermark.detection(watermarked_text)
        paraphrased_text_score = watermark.detection(p_watermarked_text)

        data = {
            'text_id': [i],
            'prompt': [prompt],
            'human': [human],
            'unwatermarked_text': [unwatermarked_text],
            'adaptive_watermarked_text': [watermarked_text],
            'paraphrased_text': [p_watermarked_text],
            'human_score': [human_score],
            'adaptive_watermarked_text_score': [adaptive_watermarked_text_score],
            'paraphrased_text_score': [paraphrased_text_score]
        }
        df  = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(f'{args.output_dir}/watermark_result.csv', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--watermark_model', default='facebook/opt-6.7b', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models. Such as mistralai/Mistral-7B-v0.1, facebook/opt-6.7b, EleutherAI/gpt-j-6b, etc.')
    parser.add_argument('--measure_model', default='gpt2-large', type=str, \
                        help='Measurement model.')
    parser.add_argument('--embedding_model', default='sentence-transformers/all-mpnet-base-v2', type=str, \
                        help='Semantic embedding model.')
    parser.add_argument('--semantic_model', default='', type=str, \
                        help='Load semantic mapping model parameters.')
    parser.add_argument('--alpha', default=2.0, type=float, \
                        help='Entropy threshold. May vary based on different measurement model. Plase select the best alpha by yourself.')
    parser.add_argument('--max_new_tokens', default=230, type=int, \
                        help='Max tokens.')
    parser.add_argument('--min_new_tokens', default=170, type=int, \
                        help='Min tokens.')
    parser.add_argument('--secret_string', default='The quick brown fox jumps over the lazy dog', type=str, \
                        help='Secret string.')
    parser.add_argument('--measure_threshold', default=50, type=float, \
                        help='Measurement threshold.')
    parser.add_argument('--delta_0', default=1.0, type=float, \
                        help='Initial Watermark Strength, which could be smaller than --delta. May vary based on different watermarking model. Plase select the best delta_0 by yourself.')
    parser.add_argument('--delta', default=1.5, type=float, \
                        help='Watermark Strength. May vary based on different watermarking model. Plase select the best delta by yourself. A excessively high delta value may cause repetition.')
    parser.add_argument('--openai_api_key', default='', type=str, \
                        help='OpenAI API key.')
    parser.add_argument('--output_dir', default='output', type=str, \
                        help='Output directory.')
    
    
    args = parser.parse_args()
    main(args)



