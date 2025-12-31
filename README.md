Optimization Experiments
Problem formulation

We frame urban traffic management as a simulation-based optimization problem.
The goal is to select policy parameters that minimize a weighted objective combining congestion and energy usage.

Decision variables

Transit frequency multiplier

Congestion toll level

Road capacity scaling factor

Objective
A weighted sum of:

system travel-time proxy

total energy usage

mean congestion

peak (95th percentile) congestion

Lower objective values indicate better overall system performance.

Method

Policies are evaluated by running a multi-modal city simulation over multiple ticks.

A random search with local refinement explores the policy space.

Each trial logs policy parameters and resulting performance metrics.

The best policy is selected based on minimum objective value.

Performance is validated against a neutral baseline policy under identical conditions.

Results

The optimizer consistently finds policies that outperform the baseline.

Key diagnostics:

Best-so-far curve shows monotonic improvement over trials.

Score histogram confirms a meaningful spread of policy quality.

Policy knob vs score plots reveal interpretable relationships between controls and outcomes.

Artifacts:

data/opt/analysis/best_so_far.png

data/opt/analysis/score_hist.png

data/opt/analysis/knobs/

The optimized policy achieves lower congestion and energy usage than baseline across evaluation runs.

Reproducibility

To reproduce results:

# run optimization
PYTHONPATH=src python src/optimization/optimizer.py

# analyze results
python src/optimization/analyze_runs.py

# evaluate baseline vs best
PYTHONPATH=src python src/optimization/evaluate_policy.py


All optimization data and plots are saved under data/opt/.

Notes

This framework is intentionally modular.
Machine learning surrogates can be added later to accelerate policy search, but are not required for optimization correctness.

3️⃣ One-paragraph project summary (for applications / portfolio)

Use this anywhere (resume bullets, project description, GitHub):

Built a simulation-based optimization framework for urban traffic systems. Policies controlling transit frequency, congestion pricing, and road capacity were optimized using black-box search over a multi-modal city simulation. Results were validated against a neutral baseline, showing consistent reductions in congestion and energy usage. The system logs structured policy–outcome data and produces diagnostic plots for interpretability and future ML integration.