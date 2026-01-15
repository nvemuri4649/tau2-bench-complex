#!/bin/bash

# Script to run trajectory generation for complex tasks only
# This generates labeled trajectories for evaluation

set -e

# Default configuration
NUM_TRIALS=${NUM_TRIALS:-1}
MAX_STEPS=${MAX_STEPS:-30}  # Complex tasks may need more steps
MAX_CONCURRENCY=${MAX_CONCURRENCY:-5}
AGENT_LLM=${AGENT_LLM:-"gpt-4o"}
USER_LLM=${USER_LLM:-"gpt-4o"}
OUTPUT_DIR=${OUTPUT_DIR:-"data/tau2/results/complex"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Running Complex Task Trajectory Generation"
echo "=========================================="
echo "Configuration:"
echo "  NUM_TRIALS: $NUM_TRIALS"
echo "  MAX_STEPS: $MAX_STEPS"
echo "  MAX_CONCURRENCY: $MAX_CONCURRENCY"
echo "  AGENT_LLM: $AGENT_LLM"
echo "  USER_LLM: $USER_LLM"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo ""

# Run telecom complex tasks
echo "=========================================="
echo "Running TELECOM Complex Tasks..."
echo "=========================================="
python -m tau2.cli run \
    --domain telecom \
    --task-split-name complex \
    --num-trials "$NUM_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --save-to "$OUTPUT_DIR/telecom_complex_${AGENT_LLM}_${NUM_TRIALS}trials"

echo ""

# Run airline complex tasks
echo "=========================================="
echo "Running AIRLINE Complex Tasks..."
echo "=========================================="
python -m tau2.cli run \
    --domain airline \
    --task-split-name complex \
    --num-trials "$NUM_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --save-to "$OUTPUT_DIR/airline_complex_${AGENT_LLM}_${NUM_TRIALS}trials"

echo ""

# Run retail complex tasks
echo "=========================================="
echo "Running RETAIL Complex Tasks..."
echo "=========================================="
python -m tau2.cli run \
    --domain retail \
    --task-split-name complex \
    --num-trials "$NUM_TRIALS" \
    --max-steps "$MAX_STEPS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --agent-llm "$AGENT_LLM" \
    --user-llm "$USER_LLM" \
    --save-to "$OUTPUT_DIR/retail_complex_${AGENT_LLM}_${NUM_TRIALS}trials"

echo ""
echo "=========================================="
echo "Complex Task Generation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results, run:"
echo "  python -m tau2.cli view <path_to_results.json>"
