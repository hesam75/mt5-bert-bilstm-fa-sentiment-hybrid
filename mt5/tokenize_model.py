import pandas as pd
import re
import unicodedata
from hazm import Normalizer
from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import argparse
import os

normalizer = Normalizer()

def normalize_farsi(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\ufeff", "")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r'[\r\n]+', ' ', text)
    text = text.replace("\u200d", "").replace("\u2060", "")
    text = text.replace("\u0640", "")
    text = re.sub(r"[يﯾﯼﯿﯽى]", "ی", text) 
    text = re.sub(r"[كک]", "ک", text)  
    text = re.sub(r"[آأإا]", "ا", text)  
    text = re.sub(r"ۀ|هٔ", "ه", text) 
    text = re.sub(r"[\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]", "", text)
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    latin_digits = "0123456789"
    trans_table = str.maketrans(persian_digits + arabic_digits, latin_digits * 2)
    text = text.translate(trans_table)

    allowed_pattern = re.compile(
        r"[^"
        r"A-Za-z" 
        r"\u0600-\u06FF" 
        r"0-9" 
        r"\s" 
        r"\.\,\!\?\:\;\u061B\u060C" 
        r"\(\)\[\]\{\}" 
        r"\"\'_%\-\+=/\\*"
        r"]"
    )
    text = allowed_pattern.sub("", text)
    text = re.sub(r"\s*([?.!,؛:،])\s*", r"\1 ", text)
    text = normalizer.normalize(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def token_count(text,tokenizer):
    tk = tokenizer.tokenize(str(text))
    return len(tk)

def tokenize_fn(batch,tokenizer,text_col,summary_col,max_input,max_target):
    model_inputs = tokenizer(
        batch[text_col], 
        max_length=max_input,
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch[summary_col], 
            max_length=max_target,
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description="Tokenize mT5")
    parser.add_argument("--min_input_tokens", type=int, default=50)
    parser.add_argument("--max_input_tokens", type=int, default=512)
    parser.add_argument("--min_target_tokens", type=int, default=20)
    parser.add_argument("--max_target_tokens", type=int, default=150)
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_csv", type=str, default="datasets/summarize.csv")
    parser.add_argument("--output_dir", type=str, default="datasets/mt5")
    parser.add_argument("--text_col", type=str, default="article")
    parser.add_argument("--summary_col", type=str, default="summary")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name,legacy=False)
    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=[args.text_col, args.summary_col])
    tqdm.pandas()
    print("normalize:")
    df[args.text_col] = df[args.text_col].progress_apply(normalize_farsi)
    df[args.summary_col] = df[args.summary_col].progress_apply(normalize_farsi)
    df = df[(df[args.text_col] != "") & (df[args.summary_col] != "")]
    df[args.text_col] ="summarize: "+df[args.text_col]
    print("tokenize:")
    df['article_tokens'] = df[args.text_col].progress_apply(lambda text:token_count(text,tokenizer))
    df['summary_tokens'] = df[args.summary_col].progress_apply(lambda text:token_count(text,tokenizer))

    df = df[
        (df['article_tokens'] >= args.min_input_tokens) & (df['article_tokens'] <= args.max_input_tokens) &
        (df['summary_tokens'] >= args.min_target_tokens) & (df['summary_tokens'] <= args.max_target_tokens)
    ].copy()

    df.drop(columns=['article_tokens', 'summary_tokens'], inplace=True)



    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed, shuffle=True)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, shuffle=True)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    valid_ds = Dataset.from_pandas(valid_df, preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df,  preserve_index=False)

    raw = DatasetDict({"train": train_ds, "validation": valid_ds, "test": test_ds})
    tokenized = raw.map(
        lambda batch: tokenize_fn(batch,tokenizer,args.text_col,args.summary_col,args.max_input_tokens, args.max_target_tokens),
        batched=True,
        remove_columns=raw["train"].column_names
    )
    tokenized.save_to_disk(args.output_dir)
    

if __name__ == "__main__":
    main()
