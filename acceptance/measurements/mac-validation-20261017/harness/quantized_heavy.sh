#!/bin/bash
# Heavy-run wrapper for the "quantized" group (scratch, not committed). The runner
# takes and releases the shared heavy-model lock around every single test.
set -u
python3 -I /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/run_quantized_gates.py /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation/quantized/binaries.json /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation/quantized "$@"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) runner exit=$?" >> /private/tmp/claude-501/-Users-nover-repos-transformers/804c4b01-7ddc-4f18-a02c-672082928230/scratchpad/mac-validation/quantized/summary.log
