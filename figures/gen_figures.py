import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BASE = '/home/alfonsoluisdelosreyes/school/198-thesis/thesis/Template__CS_199_Research_Manuscripts/figures/'

# ── Shared data ──────────────────────────────────────────────────────────────

metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'SPICE', 'METEOR']

prep_on_r32  = [48.30, 31.79, 22.13, 16.21, 36.21, 61.10, 34.22, 40.02]
prep_off_r16 = [44.95, 29.32, 20.19, 14.61, 33.73, 54.76, 31.72, 37.25]

models_full = [
    'ATHENA (Ours)', 'GPT-5', 'GPT-5-mini', 'GPT-4o',
    'GLM-4.5V-106B', 'Qwen3-VL-30B', 'Qwen3-VL-30B-Thinking',
    'Qwen2.5-VL-72B', 'GLM-4.1V-9B', 'InternVL-3.5-38B',
    'InternVL-3.5-241B', 'Gemini-2.5-Flash', 'Qwen2.5-VL-7B',
    'InternVL-3.5-1B', 'Qwen2.5-VL-3B',
]
bleu1  = [48.30, 49.17, 43.41, 40.24, 41.96, 41.07, 40.82, 40.07, 40.28, 36.94, 35.88, 33.87, 32.60, 31.73, 30.26]
bleu4  = [16.21, 14.90,  8.11,  7.97,  8.41,  8.24,  8.19,  7.90,  7.87,  7.14,  7.13,  4.83,  6.06,  6.57,  5.91]
meteor = [40.02, 27.08, 23.59, 22.98, 22.45, 23.37, 21.74, 21.15, 21.15, 20.48, 20.23, 23.01, 18.17, 17.89, 17.34]
cider  = [61.10, 66.10, 36.53, 33.10, 31.48, 31.61, 29.22, 31.24, 26.40, 26.38, 23.60, 10.21, 11.87, 11.16,  8.53]
rougel = [36.21, 34.54, 26.16, 26.79, 25.46, 25.18, 24.86, 25.14, 25.02, 24.38, 24.20, 22.53, 22.85, 22.92, 22.84]
spice  = [34.22, 35.41, 26.58, 28.73, 26.68, 27.24, 25.79, 27.40, 23.74, 26.00, 26.18, 24.53, 24.72, 24.44, 24.04]

radar_labels = ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE']
radar_models = {
    'ATHENA (Ours)': [48.30, 16.21, 40.02, 36.21, 61.10, 34.22],
    'GPT-5':         [49.17, 14.90, 27.08, 34.54, 66.10, 35.41],
    'GPT-5-mini':    [43.41,  8.11, 23.59, 26.16, 36.53, 26.58],
    'GPT-4o':        [40.24,  7.97, 22.98, 26.79, 33.10, 28.73],
}
radar_max = [49.17, 16.21, 40.02, 36.21, 66.10, 35.41]

# ── Figure 0: Preprocessing stacked comparison (Li et al. top, UWBench bottom) ─

cmp_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'SPICE', 'METEOR']

# Li et al., DoRA=False, best r per condition (scores ×100)
li_on  = [77.85, 64.51, 53.64, 44.19, 66.86, 28.45, 34.71]  # Prep ON,  r=16
li_off = [78.09, 64.67, 53.90, 44.74, 67.40, 27.90, 34.85]  # Prep OFF, r=8

# UWBench, DoRA=False, best r per condition (scores in %)
uw_on  = [48.30, 31.79, 22.13, 16.21, 36.21, 34.22, 40.02]  # Prep ON,  r=32
uw_off = [44.95, 29.32, 20.19, 14.61, 33.73, 31.72, 37.25]  # Prep OFF, r=16

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

x = np.arange(len(cmp_metrics))
w = 0.35

for ax, on_vals, off_vals, title, ylim in [
    (ax1, li_on,  li_off, '(a) Li et al. Dataset — Less Visually Degraded (Qwen3.5, DoRA=False)', 90),
    (ax2, uw_on,  uw_off, '(b) UWBench Dataset — More Visually Degraded (Qwen3.5, DoRA=False)', 57),
]:
    b1 = ax.bar(x - w/2, on_vals,  width=w, color='#2A9D8F', label='Preprocessing ON',
                zorder=3, edgecolor='white', linewidth=0.6)
    b2 = ax.bar(x + w/2, off_vals, width=w, color='#E63946', label='Preprocessing OFF',
                zorder=3, edgecolor='white', linewidth=0.6, alpha=0.85)
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7.5, color='#1a6b63', fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7.5, color='#b02030', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_metrics, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
    ax.legend(fontsize=9.5, framealpha=0.35, loc='upper right')
    ax.set_ylim(0, ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

fig.tight_layout(h_pad=3.0)
fig.savefig(BASE + 'fig_prep_gain.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_prep_gain.pdf')
plt.close()

# ── Figure 1: Preprocessing grouped bar chart ────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4.5))

x = np.arange(len(metrics))
w = 0.35

bars1 = ax.bar(x - w/2, prep_on_r32,  width=w, color='#2A9D8F', label='Preprocessing ON',
               zorder=3, edgecolor='white', linewidth=0.7)
bars2 = ax.bar(x + w/2, prep_off_r16, width=w, color='#E63946', label='Preprocessing OFF',
               zorder=3, edgecolor='white', linewidth=0.7, alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8,
            color='#1a6b63', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8,
            color='#b02030', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Preprocessing ON vs. OFF on UWBench\n(Qwen3.5, DoRA=False, best LoRA rank per condition)',
             fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=10, framealpha=0.35, loc='upper right')
ax.set_ylim(0, 74)
ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(BASE + 'fig_preprocessing_bars.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_preprocessing_bars.pdf')
plt.close()

# ── Figure 2: Radar chart ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

radar_colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
lws = [2.8, 2.0, 1.8, 1.8]
alphas = [0.18, 0.12, 0.10, 0.10]

for (name, vals), color, lw, alpha in zip(radar_models.items(), radar_colors, lws, alphas):
    norm_vals = [v / m * 100 for v, m in zip(vals, radar_max)]
    norm_vals += norm_vals[:1]
    ax.plot(angles, norm_vals, color=color, linewidth=lw, label=name, zorder=3)
    ax.fill(angles, norm_vals, color=color, alpha=alpha)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=12)
ax.set_ylim(0, 115)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(['25', '50', '75', '100'], fontsize=8, color='grey')
ax.set_title('Metric Profile: ATHENA vs. Top Frontier Models on UWBench\n(Each axis scaled to best observed score per metric)',
             fontsize=11, fontweight='bold', pad=38)
ax.legend(loc='upper right', bbox_to_anchor=(1.38, 1.05), fontsize=10, framealpha=0.4)
ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

fig.tight_layout()
fig.savefig(BASE + 'fig_radar.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_radar.pdf')
plt.close()

# ── Figure 3: Horizontal multi-metric bar chart ───────────────────────────────

order = np.argsort(bleu1)
models_s = [models_full[i] for i in order]
data_sorted = {
    'BLEU-1':  [bleu1[i]  for i in order],
    'BLEU-4':  [bleu4[i]  for i in order],
    'METEOR':  [meteor[i] for i in order],
    'ROUGE-L': [rougel[i] for i in order],
    'CIDEr':   [cider[i]  for i in order],
    'SPICE':   [spice[i]  for i in order],
}
colors_c = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#457B9D', '#6D6875']

fig, ax = plt.subplots(figsize=(11, 7))

y = np.arange(len(models_s))
h = 0.13
n_metrics = len(data_sorted)
offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * h

for (mname, vals), off, col in zip(data_sorted.items(), offsets, colors_c):
    ax.barh(y + off, vals, height=h, color=col, label=mname,
            edgecolor='white', linewidth=0.4, zorder=3)

# Highlight ATHENA
athena_idx = models_s.index('ATHENA (Ours)')
ax.axhspan(athena_idx - 0.52, athena_idx + 0.52, color='#E63946', alpha=0.07, zorder=0)
ax.text(81, athena_idx, 'ATHENA', va='center', ha='left', fontsize=9,
        color='#E63946', fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(models_s, fontsize=9.5)
ax.set_xlabel('Score (%)', fontsize=11)
ax.set_title('ATHENA vs. Frontier VLMs on UWBench-Cap (Zero-shot)\nAll Metrics, Ranked by BLEU-1',
             fontsize=11, fontweight='bold', pad=10)
ax.legend(loc='lower right', fontsize=9.5, ncol=3, framealpha=0.4)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(0, 84)

fig.tight_layout()
fig.savefig(BASE + 'fig_frontier_bars.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_frontier_bars.pdf')
plt.close()
