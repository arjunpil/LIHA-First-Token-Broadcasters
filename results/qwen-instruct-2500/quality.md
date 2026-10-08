# What the outputs that stay in the prompt language look like

Baseline outputs of non-English prompts that langdetect puts in the prompt language, split into repetition (half or more of the words repeated), prompt copy (half or more of the 4-grams taken from the prompt) and other. The last columns give the share of each kind the model's top head sends to English when removed.

| model | outputs | repetition | prompt copy | other | head | to English: repetition / copy / other |
|---|---|---|---|---|---|---|
| qwen-instruct | 1804 | 0% | 1% | 99% | L22H6 | - / 53% / 51% |
