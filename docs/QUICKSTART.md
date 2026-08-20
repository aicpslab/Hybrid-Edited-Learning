# QUICKSTART — Running the Experiments

This guide walks through every experiment script in the repository, in the order
they appear in the paper. Each section lists the script, its dependencies, the
expected outputs, and a reference runtime where one was measured.

## 1. Prerequisites

- MATLAB (R2021b or later). No additional toolboxes are required.
- Python 3 + `numpy` + `matplotlib` (optional; only for `code/tools/plot_matlab_style.py`).

## 2. One-time setup

Open MATLAB, change to the repository root, and run:

```matlab
cd C:\path\to\Hybrid-Edited-Learning
startup
```

`startup.m` does two things:

1. Changes to the repository root, so every relative output path in the
   scripts (`results/`, `fig/`) resolves to the shared folders at the root.
2. Adds the repository root, **all subfolders of `code/`**, and `misc/` to the
   MATLAB search path, so any experiment can be started by name.

> `legacy/` is deliberately **not** added to the path (it contains an older
> procedural implementation with functions of the same name, e.g.
> `legacy/taylor_expand.m`).

## 3. Output conventions

- **Results** (`.mat`, summary `.txt`) are written to the repo-root `results/`.
- **Figures** are written to the repo-root `fig/` (`fig/*.png`).
- If a script is ever run from a non-root working directory it may create a
  nested `results/` or `fig/` next to itself; those stray folders are ignored
  by `.gitignore` and can be deleted.

## 4. Experiments — Section 4.1 (Lorenz-96 dynamics)

### 4.1.a Main editing experiment
```matlab
lorenz96_experiment
```
- Reference runtime: **≈ 326 s**.
- Trains four PhNN configurations on the same Lorenz-96 data: *Unedited*,
  *PIM*, *TKM*, *PIM+TKM* (temporal), plus a quick multi-step RMSE evaluation.
- Outputs: `results/lorenz96_results.mat` (`.results .models .test_traj`),
  `fig/Lorenz96_TrainingCurves.png`, `Lorenz96_Predictions.png`,
  `Lorenz96_RMSEvsHorizon.png`, `Lorenz96_Complexity.png`,
  `Lorenz96_Weights.png`.

### 4.1.b SINDy comparison (same Taylor library, same data)
```matlab
experiment_sindy
```
- Runs standard SINDy (STLSQ) with threshold cross-validation against the PhNN
  models, then compares sparsity, precision/recall and test RMSE.
- Outputs: `results/sindy_results.mat` (`.results .models .figdata`),
  `fig/FigS1_SINDy_Coefficients.png` … `FigS5_CoefficientRecovery.png`.

```matlab
experiment_sindy_std
```
- A self-contained standard-SINDy variant with the same figures
  (`fig/FigS1` … `FigS5`). No `.mat` is saved.

### 4.1.c Controlled / quick variants
```matlab
experiment_controlled     % controlled L96 variant -> fig/Fig1..Fig6_*.png
lorenz96_quick_run        % fast demo of training + RMSE + prediction figures
```

### 4.1.d TKM validation
```matlab
tkm_validation
```
- Validates that TKM prunes exactly the cross-temporal monomials.
- Outputs: `fig/Lorenz96_TKM_Validation.png`.

### 4.1.e PCA + Maximum-Entropy hybrid framework
```matlab
hybrid_framework
```
- Runs the PCA + ME-bisecting hybrid architecture and compares N-subnetwork
  hybrids against the monolithic edited/unedited PhNNs. Optionally appends the
  TKM/PIM+TKM rows from the main experiment to its summary table.
- **Dependency:** for the TKM/PIM+TKM enrichment, run `lorenz96_experiment`
  first (the script still works without it).
- Outputs: `results/hybrid_framework_results.mat`,
  `fig/Hybrid_Scalability.png`, `Hybrid_PCA_Partitions.png`,
  `Hybrid_Pareto.png`, `Hybrid_AccuracyVsPartitions.png`,
  `Hybrid_Results.png`, `Hybrid_TrainingCost.png`.

## 5. Experiments — Sections 4.2 / 5 (coupled oscillator + control)

### 5.1 Dynamics learning
```matlab
oscillator_control
```
- Reference runtime: **≈ 94 s**.
- Trains the four PhNN configurations on a coupled oscillator network (40-D
  state + 5-D control input).
- Outputs: `results/oscillator_results.mat` (`.results`),
  `fig/FigO1_TrainingCurves.png` … `FigO5_Weights.png`.

### 5.2 Hybrid oscillator + certainty-equivalence LQR
```matlab
oscillator_hybrid_control
```
- Trains single and hybrid (subnetwork) oscillator models and evaluates
  closed-loop LQR regulation on the learned dynamics.
- Outputs: `results/oscillator_hybrid_results.mat` (default tag) and, if an
  output tag is set, `results/oscillator_hybrid_results_<TAG>.mat`;
  `fig/OscHyb_ModelAccuracy.png`, `OscHyb_ControlShooting.png`,
  `OscHyb_ControlEffect.png`, `OscHyb_ControlLQR.png`.

### 5.3 Downstream analysis (needs 5.2 first)
These scripts load `results/oscillator_hybrid_results.mat`, so run
`oscillator_hybrid_control` once before them:

```matlab
control_evaluation              % closed-loop metrics + ablation
                                %   -> results/control_results.mat, fig/FigC1..FigC4
compare_single_vs_hybrid        % single-model vs N-hybrid comparison
                                %   -> results/compare_single_vs_hybrid_<tag>.mat, fig/OscHyb_SingleVsHybrid<tag>.png
compare_single_vs_hybrid16      % fixed 16-subnetwork variant
                                %   -> results/compare_single_vs_hybrid16.mat, fig/OscHyb_SingleVsHybrid16.png
oscillator_shoot_effective      % multi-step shooting accuracy
                                %   -> results/oscillator_shoot_effective.mat, fig/OscHyb_ShootEffective.png
oscillator_shoot_long           % long-horizon shooting accuracy
                                %   -> results/oscillator_shoot_long.mat, fig/OscHyb_ShootLong.png
```

> `control_evaluation` is self-contained (it trains the models it evaluates)
> but is most meaningful after `oscillator_hybrid_control`.

## 6. Experiment — Section 5.2 (MPC path tracking)

```matlab
mpc_improved
```
- Applies the PhNN surrogate to a model-predictive-control path-tracking task
  and compares edited vs. unedited surrogates.
- Outputs: `fig/FigM1_TrainingCurves.png`, `FigM2_RMSE.png`,
  `FigM3_ParamTradeoff.png`, `FigM4_TKMvsRandom.png`.

> **Section 5.1 (AutoRally vehicle dynamics)** is reproduced with the external
> data and identification pipeline of the cited work (`mao2022phy`); no local
> code or data is included for that subsection.

## 7. Driver scripts

```matlab
run_simulations      % runs lorenz96_experiment + oscillator_control, writes
                     % results/simulation_run_log.txt + simulation_run_results.mat
save_all_results     % runs all four main experiments with error isolation
plot_all_figures     % regenerates the 12 report figures from results/*.mat
                     %   (needs lorenz96_results, sindy_results,
                     %    oscillator_results, control_results)
```

## 8. Debug / development utilities (`misc/`)

`misc/` holds utilities that are *not* part of the main pipeline:

```matlab
dump_l96hyb          % print the field structure of the three main result files
dump_results         % rerun experiments and write key metrics to results/exp_results.txt
dump_report_data     % dump saved oscillator hybrid/control data
recompute_rmse       % recompute multi-step RMSE with corrected alignment
regen_figO4          % regenerate fig/FigO4 from saved oscillator_results.mat
diag_cost, diag_shooting, verify_cost, confirm_bug, smoke_oscillator_hybrid
```

Each file's header states its purpose. These scripts expect the experiment
results to exist under `results/` (run `startup` first).

## 9. Legacy implementation (`legacy/`)

`legacy/` preserves the earlier procedural implementation of the Lorenz-96
experiment (`phnn_train.m`, `phnn_forward.m`, `generate_lorenz96_data.m`,
`build_pim_lorenz96.m`, `sindy_comparison.m`, …). It uses a different API than
the current `PhNNModel` class and is kept **for reference only**. To run it,
`cd legacy` first (it is not on the MATLAB path):

```matlab
cd legacy
main_lorenz96        % or: sindy_comparison
cd ..
```

## 10. Troubleshooting

- **"Undefined function or variable …"** — you did not run `startup`, or you
  are running a `legacy/` script without `cd legacy`. Run `startup` first.
- **Files written to a stray `results/`/`fig/` folder** — the script ran from a
  non-root working directory. `cd` to the repository root and re-run.
- **`hybrid_framework` summary table lacks the TKM/PIM+TKM rows** — run
  `lorenz96_experiment` first to produce `results/lorenz96_results.mat`.
- **Different numeric results from the paper** — training uses fixed seeds, but
  single-precision floating point and MATLAB version can still shift the last
  few digits. Trends and orders of magnitude are reproducible.
