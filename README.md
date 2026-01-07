# City Optimization Lab
Simulation-Based Policy Optimization for Urban Traffic Systems

## Overview

City Optimization Lab is a simulation-based optimization framework for urban traffic management. The system models a multi-modal city environment and searches over policy parameters to reduce congestion and energy usage.

Rather than relying on analytical gradients or closed-form assumptions, policies are evaluated directly through simulation. This allows the framework to handle complex, non-linear dynamics while remaining flexible and extensible.

The project emphasizes:
- Black-box policy optimization
- Interpretability of policy effects
- Reproducible experimentation
- A modular architecture suitable for future ML extensions

---

## Problem Formulation

Urban traffic management is framed as a simulation-driven optimization problem.

The goal is to select policy parameters that minimize a weighted objective combining congestion and energy-related metrics. Lower objective values indicate better overall system performance.

---

## Decision Variables

The optimizer searches over the following policy controls:

- **Transit frequency multiplier**  
  Scales public transit availability across the city.

- **Congestion toll level**  
  Applies pricing to road usage to influence travel demand.

- **Road capacity scaling factor**  
  Adjusts effective roadway throughput.

Each policy configuration represents a candidate intervention applied uniformly across the simulation.

---

## Objective Function

The objective is defined as a weighted sum of system-level metrics:

- System travel-time proxy  
- Total energy usage  
- Mean congestion  
- Peak congestion (95th percentile)

This formulation balances efficiency, sustainability, and robustness to extreme congestion events.

---

## Optimization Method

Policies are evaluated by running a multi-modal city simulation over multiple time steps.

The optimization loop consists of:
- Randomized policy sampling
- Local refinement around promising regions
- Logging of policy parameters and outcome metrics for every trial

The best policy is selected based on the minimum objective value observed.

Performance is validated against a neutral baseline policy evaluated under identical simulation conditions.

---

## Results

The optimizer consistently identifies policies that outperform the baseline.

Key diagnostics include:
- **Best-so-far curves** showing monotonic improvement across trials
- **Score histograms** demonstrating a meaningful spread in policy quality
- **Policy knob vs. score plots** revealing interpretable relationships between controls and outcomes

The optimized policy achieves lower congestion and reduced energy usage across evaluation runs.

---

## Artifacts

All optimization outputs are saved under:

