import argparse
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(os.path.dirname(__file__), 'bert_bilstm'))

from transformers import T5Tokenizer

from mt5.infer import summarize
from bert_bilstm.infer import sentiment,LABEL_MAP

def token_count(text,model_name):
    return len(T5Tokenizer.from_pretrained(model_name,legacy=False).tokenize(str(text)))

def main():
    parser = argparse.ArgumentParser("Inference mT5 BERT-BiLSTM")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_len", type=int, default=150)

    # mT5
    parser.add_argument("--t5_model_name", type=str, default="google/mt5-small")
    parser.add_argument("--t5_model_dir", type=str, default="models/mt5/lora-merged")
    parser.add_argument("--num_beams", type=int, default=4)
    
    
    # BERT-BiLSTM
    parser.add_argument("--bilstm_model_dir", type=str, default="models/bert_bilstm/bilstm_best.pt")
    parser.add_argument("--vocab_json", type=str, default="models/bert_bilstm//vocab.json")
    parser.add_argument("--embeddings_path", type=str, default="models/bert_bilstm//embeddings.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--bert_model_name", type=str, default="bert-base-multilingual-cased")

    parser.add_argument("--text", type=str, default=None)

    args = parser.parse_args()
    text = args.text
    if token_count(text,args.t5_model_name) > 150:
       text = summarize(text,args.t5_model_dir,args.device,args.max_len,args.num_beams)[0]
    pred_id =sentiment(text,args.bert_model_name,args.bilstm_model_dir,args.device,args.vocab_json,args.embeddings_path,args.max_len)
    print(f"sentiment: {LABEL_MAP.get(pred_id, str(pred_id))} ({pred_id})")



if __name__ == "__main__":
    main()
