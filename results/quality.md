# What the outputs that stay in the prompt language look like

Baseline outputs of non-English prompts that langdetect puts in the prompt language, split into repetition (half or more of the words repeated), prompt copy (half or more of the 4-grams taken from the prompt) and other. The last columns give the share of each kind the model's top head sends to English when removed.

| model | outputs | repetition | prompt copy | other | head | to English: repetition / copy / other |
|---|---|---|---|---|---|---|
| gpt2 | 592 | 57% | 35% | 8% | L6H10 | 89% / 82% / 85% |
| gpt2-medium | 600 | 35% | 48% | 17% | L13H6 | 75% / 81% / 81% |
| olmo2-1b | 1842 | 11% | 7% | 82% | L15H5 | 11% / 10% / 17% |
| pythia-1b | 1980 | 8% | 9% | 83% | L9H3 | 3% / 6% / 4% |
