#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config
# ----------------------------
MODELS=("mistral" "llama3" "llama3:70b")

DATE="$(date +"%Y-%m-%d_%H-%M-%S")"
OUTPUT_DIR="results_${DATE}"
mkdir -p "$OUTPUT_DIR"

METRICS_CSV="${OUTPUT_DIR}/metrics.csv"
echo "date,prompt_id,model,elapsed_s,chars,words,lines" > "$METRICS_CSV"

# Prompt suite (edit/add freely)
declare -A PROMPTS
PROMPTS["attention_5sent"]="Explain attention in transformers in 5 sentences."
PROMPTS["json_strict"]="Return ONLY valid JSON with keys: summary (string), bullets (array of 3 strings). Topic: repo specialness and why it happens."
PROMPTS["reasoning_trap"]="You have a bond with DV01=100k and convexity=12k. Rates shift +10bp then +10bp again. Explain what changes and why, without formulas."
PROMPTS["follow_instr"]="Write exactly 2 sentences. Sentence 1 must be 8 words. Sentence 2 must be 12 words. Topic: why benchmarks matter."
PROMPTS["style_hemi"]="Write a Hemingway-lite paragraph about a quiet office morning. No metaphors. No adjectives longer than 8 letters."

# ----------------------------
# Helpers
# ----------------------------

slugify () {
  # Make prompt_id safe for folder/file use (keep it simple)
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9_-' '_'
}

run_one () {
  local prompt_id="$1"
  local prompt_text="$2"
  local model="$3"
  local outdir="$4"

  local outfile="${outdir}/${model}.txt"

  # Time the full run (simple + robust)
  local start end elapsed
  start="$(date +%s)"

  # Run
  # Using argument form is cleaner than piping; both work
  ollama run "$model" "$prompt_text" > "$outfile"

  end="$(date +%s)"
  elapsed="$((end - start))"

  # Basic text stats for later analysis
  local chars words lines
  chars="$(wc -c < "$outfile" | tr -d ' ')"
  words="$(wc -w < "$outfile" | tr -d ' ')"
  lines="$(wc -l < "$outfile" | tr -d ' ')"

  echo "${DATE},${prompt_id},${model},${elapsed},${chars},${words},${lines}" >> "$METRICS_CSV"
}

# ----------------------------
# Metadata snapshot
# ----------------------------
{
  echo "date: ${DATE}"
  echo "hostname: $(hostname)"
  echo "kernel: $(uname -r)"
  echo "uptime: $(uptime)"
  echo "models:"
  printf "  - %s\n" "${MODELS[@]}"
  echo
  echo "ollama version:"
  ollama --version || true
} > "${OUTPUT_DIR}/metadata.txt"

# Optional: capture model cards / info (handy later)
mkdir -p "${OUTPUT_DIR}/model_info"
for m in "${MODELS[@]}"; do
  ollama show "$m" > "${OUTPUT_DIR}/model_info/${m}.txt" || true
done

# ----------------------------
# Benchmark run
# ----------------------------
echo "Running benchmark suite -> ${OUTPUT_DIR}"
for pid in "${!PROMPTS[@]}"; do
  prompt_text="${PROMPTS[$pid]}"
  safe_pid="$(slugify "$pid")"
  prompt_dir="${OUTPUT_DIR}/${safe_pid}"
  mkdir -p "$prompt_dir"

  # Save the prompt itself (for reproducibility)
  echo "$prompt_text" > "${prompt_dir}/prompt.txt"

  echo "Prompt: $pid"
  for model in "${MODELS[@]}"; do
    echo "  Model: $model"
    run_one "$safe_pid" "$prompt_text" "$model" "$prompt_dir"
  done
done

echo "Done. Results saved in ${OUTPUT_DIR}"
echo "Metrics: ${METRICS_CSV}"

