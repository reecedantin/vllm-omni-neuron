#!/usr/bin/env bash
# Streaming block-sparse kernel (L4, kernels/bsa_stream.py) probe -- one LNC=2 logical core (2-core lease):
#   fleet submit A14 --cores <lo>-<lo+1> --name a14-bsa-stream-probe --ram 120 --kind compile --timeout 7200 \
#     --cwd "$FLEET_WORKTREE" --raw -- "bash test/neuron/jobs/a14-bsa-stream-probe.sh"
# PROBE_VARIANTS=v1|v2|v3 picks the variant set, PROBE_SHAPES the shapes (each shape its own process so a
# failure cannot take the others down). Profiles are captured AFTER the probe processes exit (they hold the
# cores while alive), from the NEFF list the probe writes.
set -u
cd "${FLEET_WORKTREE:?}"
OUT=${SMOKE_OUT:-${FLEET_RUNS:?}/bsa-stream${PROBE_RUN:-2}}
mkdir -p "$OUT"
export PROFILE_DEFER=1
S="python -X faulthandler test/neuron/smoke_bsa_stream_trn2.py --out $OUT --reps 3 --variants ${PROBE_VARIANTS:-v2}"
rc=0
for shape in ${PROBE_SHAPES:-hunyuan fasth3-k128 fasth3}; do
  $S --shapes "$shape" --profile; r=$?
  echo "exit code: $shape=$r"
  rc=$(( rc | r ))
done
# On PATH in the Neuron SDK environment (ships next to neuron-monitor / neuron-top).
X=$(command -v neuron-explorer || echo neuron-explorer)
if [ -f "$OUT/profile_targets.txt" ]; then
  while read -r tag neff; do
    $X capture -n "$neff" -s "$OUT/profile_$tag.ntff" > "$OUT/profile_$tag.capture.log" 2>&1 \
      && $X view -n "$neff" -s "$OUT/profile_$tag.ntff" --output-format summary-text > "$OUT/profile_$tag.txt" 2>&1
    echo "profile $tag: rc=$? ($OUT/profile_$tag.txt)"
  done < "$OUT/profile_targets.txt"
fi
exit $rc
