import torch
from transformers import T5Tokenizer, MT5ForConditionalGeneration
import argparse
import re

def summarize(texts,model_dir,device_name, max_new_tokens=150, num_beams=4):
    tok = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device(device_name)
    model.to(device).eval()
    inputs = tok(texts,return_tensors="pt",padding=True,truncation=True,max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs,max_new_tokens=max_new_tokens,num_beams=num_beams,length_penalty=1.0,no_repeat_ngram_size=3)
    return [re.sub(r"<extra_id_\d+>", "", tok.decode(out, skip_special_tokens=True)).strip() for out in outputs]


def main():
    parser = argparse.ArgumentParser("Inference mT5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_dir", type=str, default="models/mt5/lora-merged") 
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()
    summary = summarize("summarize: " +args.text,args.model_dir,args.device,args.max_new_tokens,args.num_beams)[0]
    print("summarize:", summary)



if __name__ == "__main__":
    main()
