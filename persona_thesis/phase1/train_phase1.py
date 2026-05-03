import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ---- Optional cache dirs ----
os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/hf_datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/hf_cache"))

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

TRAIN_FILE = "rolebench-eng/role-generalization/general/train.jsonl"
TEST_FILE  = "rolebench-eng/role-generalization/general/test.jsonl"

TARGET_ROLES = {
    "Sheldon Cooper","Sherlock Holmes","Darth Vader","Tony Stark","Gandalf",
    "Walter White","Hermione Granger","Jack Sparrow","Yoda","Wednesday Addams"
}

def pick_answer(example):
    gen = example.get("generated", [])
    return gen[0] if isinstance(gen, list) and len(gen) > 0 else ""

def to_text(example):
    role = example["role"]
    q = example["question"]
    a = pick_answer(example)
    text = (
        f"<|system|>\nYou are role-playing as {role}. Stay in character.\n<|end|>\n"
        f"<|user|>\n{q}\n<|end|>\n"
        f"<|assistant|>\n{a}\n<|end|>\n"
    )
    return {"text": text}

def main():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- Load dataset ----
    raw = load_dataset("ZenMoore/RoleBench", data_files={"train": TRAIN_FILE, "test": TEST_FILE})
    ds = raw.filter(lambda x: x["role"] in TARGET_ROLES)
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    print(ds)
    print("Example:\n", ds["train"][0]["text"][:300])

    # ---- QLoRA 4-bit config ----
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,   # H100 supports bf16 too, but fp16 is safe
    )

    model = prepare_model_for_kbit_training(model)

    # ---- LoRA config ----
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # ---- Trainer config ----
    out_dir = os.path.abspath("./phi3_rolebench_phase1_lora")

    sft_args = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=25,
        save_steps=200,
        max_length=768,
        fp16=True,     # H100: fp16 is fine
        bf16=False,    # keep off for consistency
        report_to="none",
        dataset_text_field="text",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    print("Saved adapter to:", out_dir)

if __name__ == "__main__":
    main()