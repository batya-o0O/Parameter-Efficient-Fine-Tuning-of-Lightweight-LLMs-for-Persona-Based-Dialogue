from datasets import load_dataset

TRAIN_FILE = "rolebench-eng/role-generalization/general/train.jsonl"

raw = load_dataset("ZenMoore/RoleBench", data_files={"train": TRAIN_FILE})

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

ds = raw["train"].filter(lambda x: x["role"] in TARGET_ROLES)
ds = ds.map(to_text, remove_columns=ds.column_names)

print("Examples after filtering:", len(ds))
print("Columns:", ds.column_names)
print("\nSample:\n", ds[0]["text"][:400])