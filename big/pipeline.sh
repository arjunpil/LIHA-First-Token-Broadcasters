#!/usr/bin/env bash
# LIHA head screen for a large Qwen2.5 model, split across GPU groups.
# Called by run_32b_2gpu.sh / run_72b_4gpu.sh, which set:
#   KEY        model family key without "-instruct", e.g. qwen2.5-32b
#   GPU_GROUPS     GPU groups separated by ";", one worker per group, e.g. "0;1" or "0,1;2,3"
#   FRAC_LO/FRAC_HI  share of depth to screen (0 1 = all layers)
#   PER_LANG   prompts per language for the screen (5 languages)
#   BS         generation batch size
#   SMOKE=1    tiny run to check the pipeline end to end
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PER_LANG=${PER_LANG:-100}; BS=${BS:-64}; NLOSS=${NLOSS:-50}; TOP=${TOP:-3}
FRAC_LO=${FRAC_LO:-0}; FRAC_HI=${FRAC_HI:-1}
LCB_LANGS=${LCB_LANGS:-ar,de,en,es,fr,hi,id,it,ja,ko,pt,ru,tr,vi,zh}
LCB_EXTRA=""
INS="$KEY-instruct"
mkdir -p logs
IFS=';' read -ra G <<< "$GPU_GROUPS"
ALL_GPUS=${G[0]}   # LCB on the first group only: one GPU unless the model needs several (72B)
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a logs/progress.log; }

NL=$(python - "$INS" <<'PY'
import sys; sys.path[:0] = ["big", "experiments", "prompts"]
import run_big, sweep
from transformers import AutoConfig
print(AutoConfig.from_pretrained(sweep.MODELS[sys.argv[1]][0]).num_hidden_layers)
PY
)
LO=$(python -c "import math;print(int(math.floor($FRAC_LO*$NL)))"); HI=$(python -c "import math;print(min($NL-1,int(math.ceil($FRAC_HI*$NL))-1))")
LAYERS=($(seq "$LO" "$HI"))
if [ "${SMOKE:-0}" = "1" ]; then PER_LANG=5; NLOSS=5; LAYERS=("${LAYERS[@]:0:${#G[@]}}"); LCB_EXTRA="--limit 4"; log "SMOKE run"; fi
log "$INS: $NL layers, screening ${#LAYERS[@]} (${LAYERS[0]}..${LAYERS[-1]}) on ${#G[@]} worker(s): $GPU_GROUPS, per-lang $PER_LANG"

screen() {  # $1 = model key, rest = layers
  local key=$1; shift; local ls=("$@"); local n=${#G[@]}
  rm -rf out/"$key"-part*
  for i in "${!G[@]}"; do
    local mine=(); for j in "${!ls[@]}"; do [ $((j % n)) -eq "$i" ] && mine+=("${ls[$j]}"); done
    [ ${#mine[@]} -eq 0 ] && continue
    local csv=$(IFS=,; echo "${mine[*]}")
    log "  worker $i (GPU ${G[$i]}): $key layers $csv"
    CUDA_VISIBLE_DEVICES=${G[$i]} python big/run_big.py sweep --model "$key" --modes head \
      --per-lang "$PER_LANG" --n-loss "$NLOSS" --bs "$BS" --layers "$csv" --out "out/$key-part$i" \
      > "logs/$key-part$i.log" 2>&1 &
  done
  wait
  python big/merge_parts.py "$key" && python experiments/detect.py "$key" && python experiments/analyze.py "$key" > "logs/$key-summary.txt"
}

log "stage 1: screen $INS"
screen "$INS" "${LAYERS[@]}" || { log "stage 1 failed, see logs/"; exit 1; }
read -r TOPH TOPL < <(python big/top_heads.py "$INS" "$TOP" | paste -sd' ')
log "top correct->wrong heads in $INS: $TOPH (layers $TOPL)"

log "stage 2: same layers in the base model ($KEY)"
IFS=',' read -ra BL <<< "$TOPL"
screen "$KEY" "${BL[@]}" || log "stage 2 failed, continuing"

log "stage 3: LCB on $INS with $TOPH (zero, mean, same-layer controls)"
CUDA_VISIBLE_DEVICES=$ALL_GPUS python big/run_big.py lcb --model "$INS" --heads "$TOPH" --controls 3 \
  --langs "$LCB_LANGS" --bs 32 --token-budget 12000 --out "out/$INS-lcb" $LCB_EXTRA > "logs/$INS-lcb.log" 2>&1 \
  || log "stage 3 failed, see logs/$INS-lcb.log"

log "packing results"
tar czf "results_$KEY.tar.gz" logs out/"$KEY"* 2>/dev/null
log "done -> results_$KEY.tar.gz"
