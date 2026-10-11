# LCB-screened heads without their screen's prompts

Crosslingual paired change against base (95% bootstrap CI), on all 1,196 non-English crosslingual prompts and on
the 1,096 left after dropping the screen's 100 (screen-prompts.json; the same 100 in all five screens). Recomputed
from the saved replies with screen_heldout.py; the all-prompt values match each run's summary.json. Monolingual
prompts are not in the screens.

| model | head | run | zero, all | zero, without screen | mean, all | mean, without screen | controls, largest shift |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | L8H25 | llama3.2-1b-instruct-lcb (section 8) | -0.711 [-0.737, -0.684] | -0.711 [-0.739, -0.683] | -0.357 [-0.384, -0.329] | -0.355 [-0.383, -0.327] | 0.002 |
| Llama-3.2-3B | L13H19 | llama3.2-3b-instruct-lcb-top (section 8) | -0.335 [-0.363, -0.308] | -0.336 [-0.364, -0.308] | -0.250 [-0.277, -0.225] | -0.252 [-0.279, -0.226] | 0.001 |
| Qwen3-4B | L24H27 | qwen3-4b-instruct-lcb-top (section 8) | -0.076 [-0.093, -0.061] | -0.073 [-0.090, -0.057] | -0.040 [-0.052, -0.027] | -0.040 [-0.052, -0.026] | 0.001 |
| Qwen2.5-1.5B | L16H9 | qwen-instruct-lcbscreen-top (section 9e) | -0.459 [-0.487, -0.430] | -0.464 [-0.496, -0.435] | -0.277 [-0.303, -0.251] | -0.278 [-0.307, -0.251] | 0.001 |
| Qwen2.5-7B | L0H1 | qwen2.5-7b-instruct-lcbscreen-top (section 9e) | -0.271 [-0.300, -0.245] | -0.272 [-0.300, -0.243] | -0.005 [-0.013, +0.003] | -0.006 [-0.016, +0.002] | 0.001 |

Largest shift from dropping the screen's prompts: 0.0055 (zero), 0.0016 (mean), 0.0024 (controls).
