#!/usr/bin/env bash
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${CINN_RESOURCE_DIR:?Provide matching data/ and assets/}"
export PADDLECLAS_DIR=$(cd "$here/../../.." && pwd)
export CINN_RESOURCE_DIR
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
export PADDLECLAS_DIR
export DATA_DIR=${DATA_DIR:-$CINN_RESOURCE_DIR/data}
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
MODE=${MODE:-dynamic}
TASK=${TASK:-all}
PYTHON_BIN=${PYTHON_BIN:-python}
case "$MODE" in dynamic|cinn) ;; *) exit 2 ;; esac
case "$TASK" in train|infer|all) ;; *) exit 2 ;; esac
"$PYTHON_BIN" "$here/verify_assets.py" "$CHECKPOINT_DIR"
output_root=${RESULTS_DIR:-$here/results}
mkdir -p "$output_root"
run_task() {
    local task=$1
    local output="$output_root/$MODE-$task"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output"; exit 3; }
    local checkpoint="$CHECKPOINT_DIR/initial.pdparams"
    if [[ "$task" = infer ]]; then checkpoint="$CHECKPOINT_DIR/inference.pdparams"; fi
    "$PYTHON_BIN" "$here/run_model.py" --mode "$MODE" --task "$task" --output "$output" --checkpoint "$checkpoint" --epochs "${EPOCHS:-1}" 2>&1 | tee "$output.log"
    if [[ "$MODE" = cinn ]]; then grep -q 'Compiling subgraph with CINN backend' "$output.log" || exit 4; fi
}
if [[ "$TASK" = all ]]; then run_task train; run_task infer; else run_task "$TASK"; fi
