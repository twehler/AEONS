#!/usr/bin/env bash
# Run all three analysis scripts in sequence and report where each
# output landed.  Quiet by default — pass `-v` to see Rscript's own
# progress messages.

set -e

cd "$(dirname "$0")"

verbose=0
for arg in "$@"; do
    case "$arg" in
        -v|--verbose) verbose=1 ;;
    esac
done

run() {
    local script="$1"
    local target="$2"
    if [ "$verbose" -eq 1 ]; then
        Rscript "$script"
    else
        Rscript "$script" 2>&1 | grep -E "^(wrote|Error|Execution halted)" || true
    fi
    if [ -f "$target" ]; then
        printf "  %-44s %s\n" "$script" "→ $target"
    else
        printf "  %-44s %s\n" "$script" "(no output produced)"
    fi
}

echo "Running AEONS data-analysis suite..."
run general.R           results_general.txt
run brain_weights.R     results_brain_weights.txt
run training_curves.R   results_training_curves.txt
run limb_brains.R       results_limb_brains.txt
run correlations.R      results_correlations.txt
run time_series.R       results_time_series.txt
echo "Done."
