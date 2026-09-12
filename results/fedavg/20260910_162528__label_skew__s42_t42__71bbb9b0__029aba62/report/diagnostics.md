# Stage-2 diagnostics

Run: `20260910_162528__label_skew__s42_t42__71bbb9b0__029aba62`

## amp_skipped_steps (warning)

- Điều kiện: GradScaler skipped at least one update because gradients were non-finite
- Bằng chứng: `{"skipped_optimizer_steps": 1336}`
- Gợi ý: Inspect AMP scale and input magnitudes; compare a short FP32 run if skips persist. Optimizer steps exclude skipped updates.

## validation_plateau (info)

- Điều kiện: bad_rounds >= 3
- Bằng chứng: `{"bad_rounds": 10}`
- Gợi ý: Run long enough for the configured scheduler/early stopping; do not tune on test.

## update_norm_outlier (warning)

- Điều kiện: maximum update L2 > 5.0x median
- Bằng chứng: `{"maximum": 85.35685108915717, "median": 14.694056593795189}`
- Gợi ý: Inspect that client's data/profile and consider a lower LR in a separate run.
