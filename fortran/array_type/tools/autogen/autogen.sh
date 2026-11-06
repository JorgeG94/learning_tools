#!/bin/bash
# Regenerate the committed fypp sources under src/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
fypp "$here/devarray.fypp" "$here/../../src/devarray.F90"
echo "generated src/devarray.F90"
