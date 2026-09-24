# Configuration Files (`configs/`)

## Purpose and Scope Notice

The YAML files in this directory are **legacy test fixtures and compatibility configs** associated with the legacy `fl_training/` test suite (such as `tests/test_sweep_content_aware.py` and `tests/test_research_reporting.py`).

### Key Architectural Boundaries:
1. **`stage2_scratch` Independence**:
   - The primary Stage 2 FedAvg scratch pipeline (`stage2_scratch`, invoked via `python -m stage2_scratch <action>`) **DOES NOT** read, parse, or depend on any YAML files in this directory.
   - All runtime options for `stage2_scratch` (learning rate, batch size, epochs, seeds, rounds, paths) are explicitly configured via **CLI arguments and audited defaults** in `stage2_scratch/__main__.py` and `stage2_scratch/experiment.py`.

2. **File Inventory & Status**:
   - `train_smoke.yaml`: Minimal configuration fixture used for FL client/server initialization testing in legacy tests.
   - `stage2_controls_v4.yaml`: Configuration template for baseline experiments within the legacy `fl_training` framework.
   - `stage2_fedavg_main_v4.yaml`: Experimental parameter baseline for the legacy FL pipeline.
   - `sweep_content_aware_v4.yaml`: Parameter sweep specification across content-aware partition conditions for legacy analysis.

3. **Maintenance & Deprecation**:
   - These configs are retained strictly to preserve regression test reproducibility for legacy `fl_training` components.
   - Stage 2 scratch workflows configure parameters directly via `stage2_scratch` CLI arguments or programmatic API.
