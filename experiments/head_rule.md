# Which head the later steps use, and the Qwen2.5-7B rerun

Written 2026-10-10, 08:32 KST, after the first Qwen2.5-7B pipeline and before its layer-19 run; revised 08:33 KST,
still before that run, so that the 0.1 threshold stays on the 125-prompt screen as in section 7.

## Rule for the head that the later steps use

A model goes on to the 2,500 prompts if its strongest head reaches c->w 0.1 on the 125-prompt screen (section 7). On
the 2,500 prompts, the head that the later steps use is the one with the largest c->w among the heads that
1. have a dNLL of at most 1, and
2. keep at least 0.9 of the English prompts' continuations in English when removed.

Condition 2 is new in writing. It puts into words why SmolLM3-3B's L1H12 was set aside: its removal turned many
replies in every language, English included, into repeated `</think>` tokens (its dNLL of +2.58 also fails condition
1). It is written down now because the first Qwen2.5-7B pipeline picked L0H25, whose removal keeps only 0.51 of
English continuations in English (many become digit strings such as "001 02 03"), and the next head, L0H22, keeps
0.52 (236 of the 500 English continuations turn Chinese). The heads picked so far keep English at 1.00, so the
rule changes none of them. The threshold 0.9 sits between those values: any value from 0.6 to 0.95 picks the same
head in each of the eight instruct models with a 2,500-prompt run, Qwen2.5-7B's L0H26 included.

## Qwen2.5-7B rerun

- The first runs with L0H25 are kept under their own names (suffix -L0H25run) and reported as such.
- The screen's candidates outside layer 0 are L19H1 (0.128 on 125 prompts) and nothing else above 0.02, so layers 0
  and 19 are run on the 2,500 prompts together, and the rule above picks the head. If no head passes, Qwen2.5-7B is
  reported as having no head of this kind.
- The chosen head then gets the steps of the other models: follow-up, LCB with three same-layer controls, the
  diagnosis, steering with its quality check, and the base model on the head's layer.

## Qwen3-4B, added 2026-10-10, 08:47 KST, before its 2,500-prompt run

The Qwen3-4B pipeline was set up before the steering runs and picks its LCB head without condition 2. After it ends, the
head is picked again by the rule above. If it differs, the LCB run is kept under its own name and LCB is rerun with
the rule's head. That head then gets the diagnosis, steering and the quality check, as the other models did. If no
head passes, Qwen3-4B is reported as having no head of this kind.
