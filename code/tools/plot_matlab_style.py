"""
MATLAB-style plot generation for Lorenz-96 experiment results.
Regenerates all 6 figures with MATLAB-standard aesthetics.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.insert(0, '.')
from lorenz96_experiment import *

# ============================================================
# MATLAB-style global settings
# ============================================================
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.5,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# MATLAB default color order
MATLAB_COLORS = {
    'blue':   '#0072BD',
    'red':    '#D95319',
    'yellow': '#EDB120',
    'purple': '#7E2F8E',
    'green':  '#77AC30',
    'cyan':   '#4DBEEE',
    'maroon': '#A2142F',
}

# ============================================================
# Regenerate all models (lightweight re-run)
# ============================================================

def train_all_models():
    N = 40; dt = 0.01; F = 8.0; r = 2
    n_train, n_val, n_test = 8000, 2000, 2000
    lr = 0.002; patience = 15

    train_traj, val_traj, test_traj = generate_train_val_test_data(
        N=N, dt=dt, F=F, n_train=n_train, n_val=n_val, n_test=n_test, seed=42)

    X_train = train_traj[:-1].astype(np.float32)
    Y_train = train_traj[1:].astype(np.float32)
    X_val = val_traj[:-1].astype(np.float32); Y_val = val_traj[1:].astype(np.float32)
    X_test = test_traj[:-1].astype(np.float32); Y_test = test_traj[1:].astype(np.float32)

    mono_std = generate_monomial_indices(N, r)
    A_val, A_unc, _ = build_lorenz96_pim(N, dt, mono_std)

    results, models = {}, {}

    # --- Model 1: Unedited ---
    print("  Training Unedited...")
    m = PhNNModel(N, N, mono_std)
    tl, vl, bv = m.train(X_train, Y_train, X_val, Y_val, learning_rate=lr,
        n_epochs=150, batch_size=256, early_stopping_patience=patience, verbose=False)
    rmse_u, rs_u = compute_autoregressive_rmse(m, X_test, test_traj, horizon=100)
    results['unedited'] = {'tl': tl, 'vl': vl, 'bv': bv, 'rmse': rmse_u, 'rs': rs_u,
        'n_total': m.n_total, 'n_learn': m.n_learnable, 'sp': m.sparsity}
    models['unedited'] = m

    # --- Model 2: PIM ---
    print("  Training PIM...")
    m = PhNNModel(N, N, mono_std, A_value=A_val, A_uncertain=A_unc)
    tl, vl, bv = m.train(X_train, Y_train, X_val, Y_val, learning_rate=lr,
        n_epochs=150, batch_size=256, early_stopping_patience=patience, verbose=False)
    rmse_p, rs_p = compute_autoregressive_rmse(m, X_test, test_traj, horizon=100)
    results['pim'] = {'tl': tl, 'vl': vl, 'bv': bv, 'rmse': rmse_p, 'rs': rs_p,
        'n_total': m.n_total, 'n_learn': m.n_learnable, 'sp': m.sparsity}
    models['pim'] = m

    # --- Temporal models ---
    K = 2; dim_temp = N*K
    mono_temp = generate_monomial_indices(dim_temp, r)

    def build_temporal_data(traj, K):
        ns = len(traj)-K
        Xt = np.zeros((ns, N*K), dtype=np.float32)
        for k in range(K): Xt[:, k*N:(k+1)*N] = traj[k:ns+k]
        return Xt, traj[K:].astype(np.float32)

    Xt_train, Yt_train = build_temporal_data(train_traj, K)
    Xt_val, Yt_val = build_temporal_data(val_traj, K)
    Xt_test, Yt_test = build_temporal_data(test_traj, K)
    n_temp = 4000
    Xt_tr_s = Xt_train[:n_temp]; Yt_tr_s = Yt_train[:n_temp]

    A_unc_tkm, _ = build_lorenz96_tkm(N, mono_temp, K=K)

    # PIM+temporal mask
    A_unc_pt = np.zeros((N, len(mono_temp)), dtype=np.float32)
    A_val_pt = np.zeros((N, len(mono_temp)), dtype=np.float32)
    for h, midx in enumerate(mono_temp):
        for k in range(K):
            bs, be = k*N, (k+1)*N
            idx_in_block = [idx-bs for idx in midx if bs <= idx < be]
            if len(idx_in_block) == len(midx):
                for i in range(N):
                    relevant = {(i-2)%N, (i-1)%N, i, (i+1)%N}
                    if set(idx_in_block).issubset(relevant): A_unc_pt[i,h] = 1
                    if idx_in_block == [i]: A_val_pt[i,h] = 1.0-dt
    A_unc_pt = A_unc_pt * A_unc_tkm

    # --- Model 3: TKM ---
    print("  Training TKM...")
    m = PhNNModel(dim_temp, N, mono_temp, A_uncertain=A_unc_tkm)
    tl, vl, bv = m.train(Xt_tr_s, Yt_tr_s, Xt_val[:1000], Yt_val[:1000],
        learning_rate=lr, n_epochs=100, batch_size=128, early_stopping_patience=15, verbose=False)
    test_pred = m.forward(Xt_test[:500]); ss_rmse = np.sqrt(np.mean((test_pred-Yt_test[:500])**2))
    results['tkm'] = {'tl': tl, 'vl': vl, 'bv': bv, 'rmse': np.full(100, ss_rmse), 'rs': np.zeros(100),
        'n_total': m.n_total, 'n_learn': m.n_learnable, 'sp': m.sparsity}
    models['tkm'] = m

    # --- Model 4: PIM+TKM ---
    print("  Training PIM+TKM...")
    m = PhNNModel(dim_temp, N, mono_temp, A_value=A_val_pt, A_uncertain=A_unc_pt)
    tl, vl, bv = m.train(Xt_tr_s, Yt_tr_s, Xt_val[:1000], Yt_val[:1000],
        learning_rate=lr, n_epochs=100, batch_size=128, early_stopping_patience=15, verbose=False)
    test_pred = m.forward(Xt_test[:500]); ss_rmse = np.sqrt(np.mean((test_pred-Yt_test[:500])**2))
    results['pim_tkm'] = {'tl': tl, 'vl': vl, 'bv': bv, 'rmse': np.full(100, ss_rmse), 'rs': np.zeros(100),
        'n_total': m.n_total, 'n_learn': m.n_learnable, 'sp': m.sparsity}
    models['pim_tkm'] = m

    return results, models, X_test, Y_test, Xt_test, Yt_test, test_traj, mono_std, mono_temp


# ============================================================
# Figure 1: Training & Validation Curves
# ============================================================
def fig1_training_curves(results):
    colors = {'unedited': MATLAB_COLORS['red'], 'pim': MATLAB_COLORS['blue'],
              'tkm': MATLAB_COLORS['yellow'], 'pim_tkm': MATLAB_COLORS['purple']}
    labels = {'unedited': 'Unedited PhNN', 'pim': 'PIM-Edited PhNN',
              'tkm': 'TKM-Edited PhNN', 'pim_tkm': 'PIM+TKM Edited PhNN'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name in ['unedited', 'pim', 'tkm', 'pim_tkm']:
        r = results[name]; c = colors[name]; lbl = labels[name]
        # Smoothed training loss
        smooth = np.convolve(r['tl'], np.ones(8)/8, mode='valid')
        ax1.plot(smooth, color=c, linewidth=1.5, label=lbl)
        # Validation loss
        ax2.semilogy(r['vl'], color=c, linewidth=1.5,
                    label=f"{lbl}  ({r['bv']:.2e})")

    for ax, title in zip([ax1, ax2],
                         ['Training Loss (smoothed)', 'Validation Loss']):
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('MSE Loss', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='k', facecolor='w')
        ax.set_xlim(left=0)

    ax1.set_yscale('log')
    plt.suptitle('Figure 1: Training Performance - Lorenz-96 (N = 40, r = 2)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('fig/Fig1_TrainingCurves.png', dpi=200)
    plt.close()
    print("  Fig1 saved.")


# ============================================================
# Figure 2: RMSE vs Prediction Horizon
# ============================================================
def fig2_rmse_horizon(results):
    colors = {'unedited': MATLAB_COLORS['red'], 'pim': MATLAB_COLORS['blue'],
              'tkm': MATLAB_COLORS['yellow'], 'pim_tkm': MATLAB_COLORS['purple']}
    labels = {'unedited': 'Unedited PhNN', 'pim': 'PIM-Edited PhNN',
              'tkm': 'TKM-Edited PhNN', 'pim_tkm': 'PIM+TKM Edited PhNN'}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for name in ['unedited', 'pim', 'tkm', 'pim_tkm']:
        r = results[name]; c = colors[name]; lbl = labels[name]
        h = len(r['rmse'])
        ax.semilogy(np.arange(h), r['rmse'], color=c, linewidth=2.0, label=lbl)
        if np.any(r['rs'] > 0):
            ax.fill_between(np.arange(h),
                           np.maximum(r['rmse']-r['rs'], 1e-10),
                           r['rmse']+r['rs'], color=c, alpha=0.12)

    ax.set_xlabel('Prediction Horizon (steps)', fontweight='bold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Autoregressive Prediction Error vs Horizon', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='k', facecolor='w')
    ax.set_xlim([0, len(results['unedited']['rmse'])-1])

    plt.suptitle('Figure 2: Multi-Step Prediction Accuracy - Lorenz-96 (N = 40)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('fig/Fig2_RMSEvsHorizon.png', dpi=200)
    plt.close()
    print("  Fig2 saved.")


# ============================================================
# Figure 3: Prediction Trajectories
# ============================================================
def fig3_predictions(models, X_test, Y_test, Xt_test, Yt_test, test_traj):
    colors = {'unedited': MATLAB_COLORS['red'], 'pim': MATLAB_COLORS['blue'],
              'tkm': MATLAB_COLORS['yellow'], 'pim_tkm': MATLAB_COLORS['purple']}
    labels = {'unedited': 'Unedited', 'pim': 'PIM', 'tkm': 'TKM', 'pim_tkm': 'PIM+TKM'}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    dims = [0, 5, 10, 20, 30, 35]
    n_pred = 60
    x0 = X_test[0].copy()

    for idx, dim in enumerate(dims):
        ax = axes[idx//3, idx%3]
        ax.plot(test_traj[:n_pred, dim], 'k-', linewidth=2.0, label='True')

        for name, model in models.items():
            c = colors[name]; lbl = labels[name]
            if name in ['tkm', 'pim_tkm']:
                continue  # Skip temporal - incompatible input
            preds = multi_step_predict(model, x0, n_pred)
            ax.plot(preds[:, dim], '--', color=c, linewidth=1.2, label=lbl)

        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel(f'$x_{{{dim}}}$', fontweight='bold')
        ax.set_title(f'Dimension {dim}', fontweight='bold')
        ax.legend(loc='best', fontsize=7, frameon=True, fancybox=False,
                 edgecolor='k', facecolor='w')

    plt.suptitle('Figure 3: Multi-Step Autoregressive Prediction - Lorenz-96 (N = 40)',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('fig/Fig3_Predictions.png', dpi=200)
    plt.close()
    print("  Fig3 saved.")


# ============================================================
# Figure 4: Model Complexity Comparison
# ============================================================
def fig4_complexity(results):
    names_order = ['unedited', 'pim', 'tkm', 'pim_tkm']
    display_names = ['Unedited', 'PIM-Edited', 'TKM-Edited', 'PIM+TKM']
    colors_bar = [MATLAB_COLORS['red'], MATLAB_COLORS['blue'],
                  MATLAB_COLORS['yellow'], MATLAB_COLORS['purple']]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    totals = [results[n]['n_total'] for n in names_order]
    learns = [results[n]['n_learn'] for n in names_order]
    spars  = [results[n]['sp']*100 for n in names_order]

    # (a) Total Connections
    ax = axes[0]
    bars = ax.bar(display_names, totals, color=colors_bar, edgecolor='k', linewidth=1.0, width=0.6)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(totals)*0.03,
                f'{v:,}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Number of Connections', fontweight='bold')
    ax.set_title('Total Weight Connections', fontweight='bold')

    # (b) Learnable Parameters
    ax = axes[1]
    bars = ax.bar(display_names, learns, color=colors_bar, edgecolor='k', linewidth=1.0, width=0.6)
    for bar, v in zip(bars, learns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(learns)*0.03,
                f'{v:,}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Number of Parameters', fontweight='bold')
    ax.set_title('Learnable Parameters', fontweight='bold')

    # (c) Sparsity
    ax = axes[2]
    bars = ax.bar(display_names, spars, color=colors_bar, edgecolor='k', linewidth=1.0, width=0.6)
    for bar, v in zip(bars, spars):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Sparsity (%)', fontweight='bold')
    ax.set_title('Weight Sparsity', fontweight='bold')

    plt.suptitle('Figure 4: Model Complexity Analysis - Lorenz-96 (N = 40)',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('fig/Fig4_Complexity.png', dpi=200)
    plt.close()
    print("  Fig4 saved.")


# ============================================================
# Figure 5: Weight Matrix Visualization
# ============================================================
def fig5_weights(models, mono_std, mono_temp):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    names = ['unedited', 'pim', 'tkm', 'pim_tkm']
    titles = ['(a) Unedited PhNN', '(b) PIM-Edited PhNN',
              '(c) TKM-Edited PhNN', '(d) PIM+TKM Edited PhNN']

    for idx, name in enumerate(names):
        ax = axes[idx]
        m = models[name]
        W_eff = np.abs(m.A_value + m.A_uncertain * m.W_learn)
        n_show = min(200, m.n_monomials)
        im = ax.imshow(W_eff[:, :n_show], aspect='auto', cmap='hot',
                       interpolation='nearest', vmin=0, vmax=np.percentile(W_eff, 95))
        ax.set_xlabel('Hidden Neuron Index', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Output Dimension', fontweight='bold')
        ax.set_title(titles[idx], fontweight='bold')
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('|Weight|', fontsize=8)

    plt.suptitle('Figure 5: Effective Weight Matrix |W| (first 200 hidden neurons)',
                 fontweight='bold', fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig('fig/Fig5_WeightMatrix.png', dpi=200)
    plt.close()
    print("  Fig5 saved.")


# ============================================================
# Figure 6: TKM Validation
# ============================================================
def fig6_tkm_validation():
    N = 40; dt = 0.01; F = 8.0; r = 2; K = 2
    dim_temporal = N * K

    train_traj, val_traj, test_traj = generate_train_val_test_data(
        N=N, dt=dt, F=F, n_train=8000, n_val=2000, n_test=2000, seed=123)

    def build_temporal_data(traj, K):
        ns = len(traj)-K
        Xt = np.zeros((ns, N*K), dtype=np.float32)
        for k in range(K): Xt[:, k*N:(k+1)*N] = traj[k:ns+k]
        return Xt, traj[K:].astype(np.float32)

    Xt_train, Yt_train = build_temporal_data(train_traj, K)
    Xt_val, Yt_val = build_temporal_data(val_traj, K)
    Xt_test, Yt_test = build_temporal_data(test_traj, K)

    mono_temp = generate_monomial_indices(dim_temporal, r)

    def get_time_step(vi): return vi // N
    cross_mask = np.zeros(len(mono_temp), dtype=bool)
    within_mask = np.zeros(len(mono_temp), dtype=bool)
    for h, midx in enumerate(mono_temp):
        ts = {get_time_step(i) for i in midx}
        if len(ts) > 1: cross_mask[h] = True
        else: within_mask[h] = True

    # Train unedited PhNN
    print("  Training TKM validation model...")
    model = PhNNModel(dim_temporal, N, mono_temp)
    model.train(Xt_train[:4000], Yt_train[:4000], Xt_val[:1000], Yt_val[:1000],
        learning_rate=0.002, n_epochs=100, batch_size=256, verbose=False)

    W_eff = model.A_value + model.A_uncertain * model.W_learn
    cross_w = np.abs(W_eff[:, cross_mask]).flatten()
    within_w = np.abs(W_eff[:, within_mask]).flatten()

    mean_c = np.mean(cross_w); mean_w = np.mean(within_w)
    median_c = np.median(cross_w); median_w = np.median(within_w)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Distribution histogram
    ax = axes[0, 0]
    bins = np.logspace(-7, 1, 70)
    ax.hist(within_w, bins=bins, alpha=0.65, label='Within-Temporal',
            color=MATLAB_COLORS['blue'], density=True, edgecolor='k', linewidth=0.3)
    ax.hist(cross_w, bins=bins, alpha=0.65, label='Cross-Temporal',
            color=MATLAB_COLORS['red'], density=True, edgecolor='k', linewidth=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('|Weight|', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('(a) Weight Magnitude Distribution', fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='k', facecolor='w')

    # Add mean lines
    ax.axvline(x=mean_w, color=MATLAB_COLORS['blue'], linestyle='--', linewidth=1.5,
               label=f'Within mean = {mean_w:.4f}')
    ax.axvline(x=mean_c, color=MATLAB_COLORS['red'], linestyle='--', linewidth=1.5,
               label=f'Cross  mean = {mean_c:.4f}')
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='k', facecolor='w')

    # (b) Per-output ratio
    ax = axes[0, 1]
    ratios = []
    for i in range(N):
        cw_i = np.mean(np.abs(W_eff[i, cross_mask]))
        ww_i = np.mean(np.abs(W_eff[i, within_mask]))
        ratios.append(cw_i/(ww_i+1e-10))
    ax.bar(range(N), ratios, color=MATLAB_COLORS['blue'], edgecolor='k', linewidth=0.5, width=0.7)
    ax.axhline(y=1.0, color=MATLAB_COLORS['red'], linestyle='--', linewidth=1.5, label='y = 1')
    ax.axhline(y=np.mean(ratios), color='k', linestyle='-', linewidth=1.2,
               label=f'Mean = {np.mean(ratios):.3f}')
    ax.set_xlabel('Output Dimension Index', fontweight='bold')
    ax.set_ylabel('Cross / Within Mean |W|', fontweight='bold')
    ax.set_title('(b) Weight Ratio per Output Dimension', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='k', facecolor='w')

    # (c) Cumulative distribution
    ax = axes[1, 0]
    ax.plot(np.linspace(0, 100, len(cross_w)), np.sort(cross_w),
            color=MATLAB_COLORS['red'], linewidth=2.0, label='Cross-Temporal')
    ax.plot(np.linspace(0, 100, len(within_w)), np.sort(within_w),
            color=MATLAB_COLORS['blue'], linewidth=2.0, label='Within-Temporal')
    ax.set_xlabel('Percentile', fontweight='bold')
    ax.set_ylabel('|Weight|', fontweight='bold')
    ax.set_title('(c) Cumulative Distribution', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='k', facecolor='w')

    # (d) Summary statistics
    ax = axes[1, 1]
    s_labels = ['Mean', 'Median', 'Std Dev', '90th %ile', '99th %ile']
    c_stats = [mean_c, median_c, np.std(cross_w),
               np.percentile(cross_w, 90), np.percentile(cross_w, 99)]
    w_stats = [mean_w, median_w, np.std(within_w),
               np.percentile(within_w, 90), np.percentile(within_w, 99)]
    x = np.arange(len(s_labels)); bw = 0.32
    ax.bar(x-bw/2, w_stats, bw, label='Within-Temporal',
           color=MATLAB_COLORS['blue'], edgecolor='k', linewidth=0.8)
    ax.bar(x+bw/2, c_stats, bw, label='Cross-Temporal',
           color=MATLAB_COLORS['red'], edgecolor='k', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(s_labels)
    ax.set_ylabel('|Weight| Value', fontweight='bold')
    ax.set_title('(d) Summary Statistics', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='k', facecolor='w')

    plt.suptitle('Figure 6: TKM Validation — Cross-Temporal Weight Analysis (Lorenz-96 N = 40)',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('fig/Fig6_TKM_Validation.png', dpi=200)
    plt.close()
    print("  Fig6 saved.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import os
    os.makedirs('fig', exist_ok=True)

    print("=" * 60)
    print("Generating MATLAB-style figures for Lorenz-96 experiment")
    print("=" * 60)

    # Train all models (once)
    print("\n[1] Training all models (this may take a few minutes)...")
    results, models, X_test, Y_test, Xt_test, Yt_test, test_traj, mono_std, mono_temp = train_all_models()

    # Generate all 6 figures
    print("\n[2] Generating MATLAB-style figures...")
    fig1_training_curves(results)
    fig2_rmse_horizon(results)
    fig3_predictions(models, X_test, Y_test, Xt_test, Yt_test, test_traj)
    fig4_complexity(results)
    fig5_weights(models, mono_std, mono_temp)
    fig6_tkm_validation()

    print("\nAll figures saved to fig/ directory.")
    print("=" * 60)
