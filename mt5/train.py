from datasets import load_from_disk
from transformers import (
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    MT5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,GenerationConfig
)
import argparse
import numpy as np
import evaluate
import pandas as pd
from transformers.trainer_utils import get_last_checkpoint
import os
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tokenize_model import normalize_farsi
rouge = evaluate.load("rouge")
def compute_metrics(eval_preds,tok):
    preds, labels = eval_preds
    if isinstance(preds, tuple): 
        preds = preds[0]

    decoded_preds = tok.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tok.pad_token_id)
    decoded_labels = tok.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [normalize_farsi(x) for x in decoded_preds]
    decoded_labels = [normalize_farsi(x) for x in decoded_labels]

    res = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=False,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    )
    return {k: round(v * 100, 2) for k, v in res.items()} 




def main():
    parser = argparse.ArgumentParser(description="Train mT5")
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenized_dir", type=str, default="datasets/mt5")
    parser.add_argument("--output_dir", type=str, default="models/mt5")
    parser.add_argument("--log_dir", type=str, default="logs/mt5")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)

    ds = load_from_disk(args.tokenized_dir)

    tok = T5Tokenizer.from_pretrained(args.model_name, legacy=False)
    base_model = MT5ForConditionalGeneration.from_pretrained(args.model_name,   trust_remote_code=True,use_safetensors=True)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        r=16, 
        lora_alpha=32,  
        lora_dropout=0.05, 
        target_modules=["q", "v"],  # آموزش سنگین تر ["q","k","v","o"]
    )

    model = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False
    model.enable_input_require_grads()

    model.print_trainable_parameters() 

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, pad_to_multiple_of=8, padding=True)

    
    gen_config = GenerationConfig(max_length=150, num_beams=4, length_penalty=1.0)
        
    trainArgs = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        overwrite_output_dir=True,
        seed=args.seed,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
        optim="adafactor", 
        lr_scheduler_type="linear", 
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=10,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="none",
        generation_config=gen_config,
        num_train_epochs=args.epochs, 
        max_grad_norm=0.8
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=trainArgs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=lambda x:compute_metrics(x, tok),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(
        resume_from_checkpoint=(
            get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        )
    )


    test_metrics = trainer.evaluate(eval_dataset=ds["test"])
    print("Saving Log!!")
    pd.DataFrame([test_metrics]).to_csv(os.path.join(args.log_dir, "test_metrics.csv"), index=False, encoding="utf-8")
    pd.DataFrame(trainer.state.log_history).to_csv( os.path.join(args.log_dir, "log_history.csv"), index=False, encoding="utf-8-sig")
        
    final_dir =os.path.join(args.output_dir, "LoRA")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)  
    tok.save_pretrained(final_dir)

    try:
        base_for_merge = MT5ForConditionalGeneration.from_pretrained(args.model_name)
        peft_for_merge = PeftModel.from_pretrained(base_for_merge, final_dir)
        merged = peft_for_merge.merge_and_unload() 

        merged_dir = os.path.join(args.output_dir, "LoRA-merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print("Merged model saved to:", merged_dir)
    except Exception as e:
        print("Merging skipped (optional):", e)




if __name__ == "__main__":
    main()
