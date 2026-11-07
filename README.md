> **Attribution:** The original base demo in this repository was provided by [SOURCE] and is preserved unmodified in the initial commit. All extensions and later changes are mine and are documented in the commit history.

# Mini Sleep-Cycle Demo

A student-scale reproduction of the core idea from Google's paper
**"Language Models Need Sleep"** (Behrouz et al., 2026) — a language model that
learns new facts during a **wake** phase, then consolidates them into its own
weights during an offline **sleep** phase, instead of forgetting everything
when the chat ends.

> Not affiliated with Google. The official paper ships no code; this is an
> independent, small-scale re-creation of the *recipe* for learning purposes.

## The idea in one paragraph

Normal LLMs are frozen after training: anything they "learn" in a conversation
lives in the context window and evaporates. The Sleep paper proposes a cycle —
**wake** (new information sits in fast, temporary memory), then **sleep** in
two stages: **consolidation** (distill the temporary knowledge into durable
weights) and **dreaming** (the model generates its own synthetic practice
examples and trains on them, with replay of old knowledge mixed in so it
doesn't catastrophically forget). This demo runs that exact cycle on a small
open model you can train on a laptop or a free Colab GPU.

## What the demo does

1. **Baseline** — Ask the base model 20 questions about a fictional planet it
   has never heard of. It scores ~0. It also takes a general-knowledge quiz
   (the "don't forget this" control).
2. **Wake** — Give the model the facts *in context* and ask again. It scores
   high — but this knowledge is temporary (context only).
3. **Dream** — The model writes its own practice Q&A pairs about the new
   facts (self-generated rehearsal data), mixed with general-knowledge
   replay examples.
4. **Sleep** — LoRA fine-tune the model on the dream data. This is the
   consolidation step: knowledge moves from context into weights.
5. **Morning after** — Ask the fictional-planet questions again *with no
   context*. Whatever it now answers correctly lives in its weights.
   Re-take the general quiz to measure how much old knowledge survived
   (the forgetting check).

Results print as a score table and are saved to `results.md`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python sleep_demo.py
```

Options:

```bash
python sleep_demo.py --model Qwen/Qwen2.5-1.5B-Instruct   # default
python sleep_demo.py --model Qwen/Qwen2.5-3B-Instruct     # stronger, slower
python sleep_demo.py --epochs 3                            # sleep longer
python sleep_demo.py --skip-dreams                         # train on facts directly (ablation)
```

First run downloads the model (~3 GB for the 1.5B). A full cycle takes
roughly 5–15 minutes on an Apple Silicon laptop or a free Colab T4, longer
on CPU only.

## How it maps to the paper

| Paper concept | This demo |
|---|---|
| Wake phase / fast memory | Facts provided in the prompt (context window) |
| Knowledge seeding / consolidation | LoRA fine-tune on fact-derived Q&A |
| Dreaming (synthetic self-training) | Model generates its own practice Q&A pairs |
| Rehearsal against forgetting | General-knowledge examples mixed into training |
| Catastrophic forgetting check | General quiz scored before vs. after sleep |
| Hope multi-rate memory | *Not reproduced* — no public code exists; a single LoRA adapter plays the role of the "new memory block" |

Honest limits: the paper used custom architectures and RL-based dreaming
with gradient-scored example selection. This demo keeps the *structure*
(wake → dream → consolidate → measure forgetting) with commodity tools —
which is exactly what makes it a good learning project.

## Files

- `sleep_demo.py` — the whole pipeline, commented phase by phase
- `facts.json` — 20 invented facts about the fictional planet Thalassa-9,
  each with a test question and expected answer keywords
- `general_questions.json` — 15 general-knowledge questions (forgetting control)
- `requirements.txt` — torch, transformers, peft, accelerate
- `results.md` — generated after a run

## Ideas to extend it 

- Swap the dream generator's parsing for structured JSON output
- Add a second sleep cycle with a *second* fact set and measure interference
- Compare LoRA ranks (r=4 vs r=16) on retention vs. forgetting
- Replace keyword scoring with an LLM-judge eval
- Try the same cycle on the 3B model and plot the difference
