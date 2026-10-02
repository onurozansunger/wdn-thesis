# Experimental evidence

This directory contains 218 experimental records, retained byte for byte. [manifest.json](manifest.json) lists their original paths, byte counts and SHA-256 hashes. The [evidence register](../../docs/evidence-register.md) maps the dissertation’s Appendix A identifiers E1–E9 to these files. Its machine-readable form is [register.json](register.json).

From the repository root, run:

```bash
python3 scripts/verify_release.py
python3 results/evidence/verify.py
```

The evidence verifier checks every record hash, the original thirty-pair replay confirmation, the supplementary single-classifier confusion counts, means and thresholds, and the historical zero- versus three-hour comparison. The release verifier additionally checks all 120 final evaluation cells and the E1–E9 file map. Both use the Python standard library.

The supplementary baseline records include the frozen protocol, complete per-source results and per-model settings. Both information allowances and both calibration objectives are retained. These evaluations reuse six sources per network after the primary hybrid results were known; they are a supplementary controlled comparison. The historical latency comparison concerns the earlier Modena configuration.

The records support inspection of calculations. Large simulation arrays, prediction arrays and fitted model binaries remain outside this repository. Their identities are recorded in the artifact audit; they are required to regenerate predictions. See [REPRODUCE.md](../../REPRODUCE.md) for the scope of verification and full rerun requirements.
