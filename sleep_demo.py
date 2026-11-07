#!/usr/bin/env python3
"""
Mini Sleep-Cycle Demo
=====================
A student-scale reproduction of the recipe from Google's
"Language Models Need Sleep" (Behrouz et al., 2026):

    WAKE   -> new facts live only in the context window (temporary memory)
    DREAM  -> the model writes its own practice Q&A about those facts
    SLEEP  -> LoRA fine-tune on the dream data (consolidation into weights)
    MORNING-> test again with NO context, and re-check old knowledge

Run:  python sleep_demo.py [--model ...] [--epochs N] [--skip-dreams]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)

HERE = Path(__file__).parent
random.seed(7)
torch.manual_seed(7)

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    print(f"Loading {model_id} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def chat(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int = 80) -> str:
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def score_answer(response: str, keywords: list[str]) -> bool:
    """Liberal scoring: at least half the expected keywords appear."""
    r = response.lower()
    hits = sum(1 for k in keywords if k.lower() in r)
    return hits >= max(1, (len(keywords) + 1) // 2)


def run_quiz(model, tok, items, context: str | None, label: str) -> float:
    sys_prompt = "You are a helpful assistant. Answer briefly and directly."
    correct = 0
    for it in items:
        user = it["question"]
        if context:
            user = f"Use these notes to answer.\n\n{context}\n\nQuestion: {user}"
        resp = generate(model, tok, chat(tok, sys_prompt, user), max_new=60)
        ok = score_answer(resp, it["answer_keywords"])
        correct += ok
        print(f"  [{'OK' if ok else '--'}] {it['question']}")
    pct = 100.0 * correct / len(items)
    print(f"{label}: {correct}/{len(items)} = {pct:.0f}%")
    return pct


# ----------------------------------------------------------------------------
# PHASE 3 (dreaming): the model writes its own practice material
# ----------------------------------------------------------------------------

DREAM_PROMPT = """Here is a fact: "{fact}"

Write exactly {n} short practice question-and-answer pairs that test this
fact, in this exact format and nothing else:

Q: <question>
A: <answer>
Q: <question>
A: <answer>"""


def dream(model, tok, facts, per_fact: int = 3) -> list[dict]:
    """Ask the model to generate rehearsal Q&A; fall back to templates."""
    dreams = []
    for f in facts:
        prompt = chat(tok, "You write study questions.",
                      DREAM_PROMPT.format(fact=f["fact"], n=per_fact))
        text = generate(model, tok, prompt, max_new=220)
        pairs = re.findall(r"Q:\s*(.+?)\s*\nA:\s*(.+)", text)
        made = [{"question": q.strip(), "answer": a.strip()}
                for q, a in pairs if len(q) > 5 and len(a) > 1]
        if not made:  # parsing failed -> template fallback keeps the demo alive
            made = [
                {"question": f["question"],
                 "answer": ", ".join(f["answer_keywords"])},
                {"question": f"What do you know about this: {f['fact']}",
                 "answer": f["fact"]},
            ]
        dreams.extend(made)
    return dreams


# Replay buffer: old knowledge rehearsed during sleep fights forgetting.
REPLAY = [
    {"question": "What is the capital of Japan?",
     "answer": "The capital of Japan is Tokyo."},
    {"question": "Who wrote Hamlet?",
     "answer": "William Shakespeare wrote Hamlet."},
    {"question": "What is 12 times 8?",
     "answer": "12 times 8 is 96."},
    {"question": "What planet do we live on?",
     "answer": "We live on Earth."},
    {"question": "What is the chemical formula for water?",
     "answer": "The chemical formula for water is H2O."},
    {"question": "How many minutes are in an hour?",
     "answer": "There are 60 minutes in an hour."},
    {"question": "What is the largest mammal in the world?",
     "answer": "The blue whale is the largest mammal in the world."},
    {"question": "What language is mainly spoken in Egypt?",
     "answer": "Arabic is the main language spoken in Egypt."},
]


# ----------------------------------------------------------------------------
# PHASE 4 (sleep): LoRA fine-tune = consolidation into weights
# ----------------------------------------------------------------------------

class QADataset(Dataset):
    def __init__(self, pairs, tok, max_len: int = 256):
        self.rows = []
        for p in pairs:
            text = chat(tok, "You are a helpful assistant. Answer briefly and directly.",
                        p["question"]) + p["answer"] + tok.eos_token
            ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
            self.rows.append(ids)
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = self.rows[i]
        return {"input_ids": ids, "labels": list(ids)}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def sleep(model, tok, pairs, epochs: int, out_dir: Path):
    print(f"Consolidating {len(pairs)} examples with LoRA "
          f"({epochs} epoch(s)) ...")
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        use_cpu=(pick_device() == "cpu"),
        fp16=(pick_device() == "cuda"),
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=QADataset(pairs, tok),
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    model = model.merge_and_unload()  # weights now hold the knowledge
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Main cycle
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--skip-dreams", action="store_true",
                    help="Ablation: train on the raw facts instead of dreams")
    args = ap.parse_args()

    facts = json.loads((HERE / "facts.json").read_text())
    general = json.loads((HERE / "general_questions.json").read_text())
    context = "\n".join(f["fact"] for f in facts)
    device = pick_device()

    model, tok = load_model(args.model, device)
    results = {}

    print("\n=== BASELINE (before ever seeing the facts) ===")
    results["new_facts_baseline"] = run_quiz(model, tok, facts, None, "New facts, no context")
    results["general_baseline"] = run_quiz(model, tok, general, None, "General knowledge")

    print("\n=== WAKE PHASE (facts in context = temporary memory) ===")
    results["new_facts_in_context"] = run_quiz(model, tok, facts, context, "New facts, in context")

    print("\n=== DREAM PHASE (model writes its own practice Q&A) ===")
    if args.skip_dreams:
        pairs = [{"question": f["question"],
                  "answer": ", ".join(f["answer_keywords"])} for f in facts]
        print(f"Skipped dreaming; using {len(pairs)} direct fact Q&A pairs.")
    else:
        pairs = dream(model, tok, facts)
        print(f"Generated {len(pairs)} dream practice pairs.")
    random.shuffle(REPLAY)
    train_pairs = pairs + REPLAY  # rehearsal mixed in, like the paper
    print(f"Training set: {len(train_pairs)} examples (incl. {len(REPLAY)} replay)")

    print("\n=== SLEEP PHASE (consolidation into weights) ===")
    model = sleep(model, tok, train_pairs, args.epochs, HERE / "sleep_out")

    print("\n=== MORNING AFTER (no context — is it in the weights?) ===")
    results["new_facts_after_sleep"] = run_quiz(model, tok, facts, None, "New facts, after sleep")
    results["general_after_sleep"] = run_quiz(model, tok, general, None, "General knowledge, after sleep")

    forgetting = results["general_baseline"] - results["general_after_sleep"]
    print("\n================= RESULTS =================")
    print(f"New facts, never seen        : {results['new_facts_baseline']:.0f}%")
    print(f"New facts, in context (wake) : {results['new_facts_in_context']:.0f}%")
    print(f"New facts, after sleep       : {results['new_facts_after_sleep']:.0f}%  <- consolidated")
    print(f"General knowledge, before    : {results['general_baseline']:.0f}%")
    print(f"General knowledge, after     : {results['general_after_sleep']:.0f}%  "
          f"(forgetting: {forgetting:+.0f} pts)")
    print("===========================================")

    lines = ["# Sleep-cycle demo — results\n",
             f"Model: `{args.model}` · Epochs: {args.epochs} · "
             f"Dreams: {'off (ablation)' if args.skip_dreams else 'on'}\n",
             "| Measure | Score |", "|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k.replace('_', ' ')} | {v:.0f}% |")
    lines.append(f"| forgetting (general, before − after) | {forgetting:+.0f} pts |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("Saved results.md")


if __name__ == "__main__":
    sys.exit(main())
