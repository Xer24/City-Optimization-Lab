# Optimization Run Analysis

- Runs analyzed: **278**

- Best score: **693843** (run_id=173)

- Best policy JSON: `data/opt/best_policy.json`


## Plots

- Best-so-far curve: `data/opt/analysis/best_so_far.png`

- Score histogram: `data/opt/analysis/score_hist.png`

- Knob relationship plots folder: `data/opt/analysis/knobs`


## Best policy parameters (from best run row)

```json

{
  "transit_frequency_mult":0.9511146958,
  "congestion_toll":3.7134628336,
  "road_capacity_scale":0.8058731374
}

```


## Knob correlations with score (Pearson)

| knob                   |   pearson_corr_with_score |
|:-----------------------|--------------------------:|
| road_capacity_scale    |                -0.23179   |
| congestion_toll        |                -0.0301361 |
| transit_frequency_mult |                -0.0257233 |



## Notes

- Lower score is better.

- Correlation does not prove causation; use as a directional hint.
