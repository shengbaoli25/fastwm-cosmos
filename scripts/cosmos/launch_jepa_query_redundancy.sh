#!/usr/bin/env bash
# Same selected encoder and B64/30k delta recipe, with weak spatial-query regularization.
set -euo pipefail
JEPA_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$JEPA_SOURCE/scripts/cosmos/launch_jepa_selected.sh" \
  --query-redundancy-recipe "$JEPA_SOURCE/configs/experiments/jepa_query_redundancy.json" "$@"
