import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Data ────────────────────────────────────────────────────────────────────

metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'SPICE', 'METEOR']

# Preprocessing ON/OFF — DoRA=True, all LoRA ranks (Qwen3.5 on UWBench)
prep_on_dora_true = {
    'r=8':  [48.83, 31.95, 22.12, 16.14, 35.94, 59.16, 34.17, 40.27],
    'r=16': [47.25, 31.08, 21.67, 15.88, 35.93, 59.44, 33.79, 39.24],
    'r=32': [48.51, 31.85, 22.17, 16.24, 36.18, 60.05, 34.30, 40.13],
}
prep_off_dora_true = {
    'r=8':  [34.33, 22.18, 15.14, 10.89, 25.93, 43.21, 24.54, 29.26],
    'r=16': [35.55, 22.98, 15.72, 11.32, 26.15, 38.41, 24.80, 29.24],
    'r=32': [33.94, 21.96, 15.03, 10.83, 25.45, 38.91, 24.37, 28.81],
}

# ATHENA vs frontier models on UWBench
models = [
    'ATHENA\n(Ours)', 'GPT-5', 'GPT-5-mini', 'GPT-4o',
    'GLM-4.5V\n106B', 'Qwen3-VL\n30B', 'Qwen3-VL\n30B-Think',
    'Qwen2.5-VL\n72B', 'GLM-4.1V\n9B', 'InternVL\n3.5-38B',
    'InternVL\n3.5-241B', 'Gemini\n2.5-Flash', 'Qwen2.5-VL\n7B',
    'InternVL\n3.5-1B', 'Qwen2.5-VL\n3B',
]
bleu1  = [48.30, 49.17, 43.41, 40.24, 41.96, 41.07, 40.82, 40.07, 40.28, 36.94, 35.88, 33.87, 32.60, 31.73, 30.26]
bleu4  = [16.21, 14.90,  8.11,  7.97,  8.41,  8.24,  8.19,  7.90,  7.87,  7.14,  7.13,  4.83,  6.06,  6.57,  5.91]
meteor = [40.02, 27.08, 23.59, 22.98, 22.45, 23.37, 21.74, 21.15, 21.15, 20.48, 20.23, 23.01, 18.17, 17.89, 17.34]
cider  = [61.10, 66.10, 36.53, 33.10, 31.48, 31.61, 29.22, 31.24, 26.40, 26.38, 23.60, 10.21, 11.87, 11.16,  8.53]
rougel = [36.21, 34.54, 26.16, 26.79, 25.46, 25.18, 24.86, 25.14, 25.02, 24.38, 24.20, 22.53, 22.85, 22.92, 22.84]
spice  = [34.22, 35.41, 26.58, 28.73, 26.68, 27.24, 25.79, 27.40, 23.74, 26.00, 26.18, 24.53, 24.72, 24.44, 24.04]

# Radar data: ATHENA vs top-3 competitors
radar_labels = ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE']
radar_models = {
    'ATHENA (Ours)': [48.30, 16.21, 40.02, 36.21, 61.10, 34.22],
    'GPT-5':         [49.17, 14.90, 27.08, 34.54, 66.10, 35.41],
    'GPT-5-mini':    [43.41,  8.11, 23.59, 26.16, 36.53, 26.58],
    'GPT-4o':        [40.24,  7.97, 22.98, 26.79, 33.10, 28.73],
}
# Normalize to 0-100 scale for radar (max observed per metric)
radar_max = [49.17, 16.21, 40.02, 36.21, 66.10, 35.41]

# ── Figure layout ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 16))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653',
           '#6D6875', '#B5838D', '#FFCB77', '#5E548E']

# ── Panel A: Preprocessing ON vs OFF grouped bar (best per condition) ────────
ax_a = fig.add_subplot(gs[0, 0])

x = np.arange(len(metrics))
w = 0.35

on_vals  = [48.30, 31.79, 22.13, 16.21, 36.21, 61.10, 34.22, 40.02]  # DoRA=False, r=32, Prep=ON
off_vals = [44.95, 29.32, 20.19, 14.61, 33.73, 54.76, 31.72, 37.25]  # DoRA=False, r=16, Prep=OFF

bars1 = ax_a.bar(x - w/2, on_vals,  width=w, color='#2A9D8F', label='Preprocessing ON',  zorder=3, edgecolor='white', linewidth=0.6)
bars2 = ax_a.bar(x + w/2, off_vals, width=w, color='#E63946', label='Preprocessing OFF', zorder=3, edgecolor='white', linewidth=0.6, alpha=0.85)

for bar in bars1:
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6.5, color='#2A9D8F', fontweight='bold')
for bar in bars2:
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6.5, color='#E63946', fontweight='bold')

ax_a.set_xticks(x)
ax_a.set_xticklabels(metrics, fontsize=9)
ax_a.set_ylabel('Score (%)', fontsize=10)
ax_a.set_title('(a) Preprocessing Impact on UWBench\n(Qwen3.5, DoRA=False, best per condition)', fontsize=10, fontweight='bold', pad=8)
ax_a.legend(fontsize=9, framealpha=0.4)
ax_a.set_ylim(0, 72)
ax_a.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax_a.set_axisbelow(True)

# Shade gap between ON and OFF
for i, (on, off) in enumerate(zip(on_vals, off_vals)):
    ax_a.annotate('', xy=(x[i]+w/2, off), xytext=(x[i]-w/2, on),
                  arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.8))

# ── Panel B: Radar chart ─────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1], polar=True)

N = len(radar_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

radar_colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']
for (name, vals), color in zip(radar_models.items(), radar_colors):
    norm_vals = [v / m * 100 for v, m in zip(vals, radar_max)]
    norm_vals += norm_vals[:1]
    ax_b.plot(angles, norm_vals, color=color, linewidth=2, label=name)
    ax_b.fill(angles, norm_vals, color=color, alpha=0.12)

ax_b.set_xticks(angles[:-1])
ax_b.set_xticklabels(radar_labels, fontsize=9.5)
ax_b.set_ylim(0, 110)
ax_b.set_yticks([25, 50, 75, 100])
ax_b.set_yticklabels(['25', '50', '75', '100'], fontsize=7, color='grey')
ax_b.set_title('(b) Metric Profile vs. Frontier Models\n(Normalized to observed max)', fontsize=10, fontweight='bold', pad=20)
ax_b.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8.5, framealpha=0.4)
ax_b.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

# ── Panel C: ATHENA vs all models — horizontal bar (BLEU-1 ranked) ───────────
ax_c = fig.add_subplot(gs[1, :])

order = np.argsort(bleu1)
models_s = [models[i] for i in order]
bleu1_s  = [bleu1[i]  for i in order]
bleu4_s  = [bleu4[i]  for i in order]
meteor_s = [meteor[i] for i in order]
cider_s  = [cider[i]  for i in order]
rougel_s = [rougel[i] for i in order]
spice_s  = [spice[i]  for i in order]

y = np.arange(len(models_s))
h = 0.13
offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * h

metric_names = ['BLEU-1', 'BLEU-4', 'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE']
data_sets    = [bleu1_s, bleu4_s, meteor_s, rougel_s, cider_s, spice_s]
colors_c     = ['#E63946', '#F4A261', '#E9C46A', '#2A9D8F', '#457B9D', '#6D6875']

for (off, vals, col, mname) in zip(offsets, data_sets, colors_c, metric_names):
    bars = ax_c.barh(y + off, vals, height=h, color=col, label=mname,
                     edgecolor='white', linewidth=0.4, zorder=3)

# Highlight ATHENA row
athena_idx = models_s.index('ATHENA\n(Ours)')
ax_c.axhspan(athena_idx - 0.5, athena_idx + 0.5, color='#E63946', alpha=0.06, zorder=0)
ax_c.axvline(x=0, color='grey', linewidth=0.5)

ax_c.set_yticks(y)
ax_c.set_yticklabels(models_s, fontsize=8.5)
ax_c.set_xlabel('Score (%)', fontsize=10)
ax_c.set_title('(c) ATHENA vs. Frontier VLMs on UWBench-Cap (Zero-shot) — All Metrics, Sorted by BLEU-1',
               fontsize=10, fontweight='bold', pad=8)
ax_c.legend(loc='lower right', fontsize=8.5, ncol=6, framealpha=0.4)
ax_c.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax_c.set_axisbelow(True)
ax_c.set_xlim(0, 82)

fig.suptitle('ATHENA: Supplemental Performance Analysis', fontsize=13, fontweight='bold', y=0.99)

out = '/home/alfonsoluisdelosreyes/school/198-thesis/thesis/Template__CS_199_Research_Manuscripts/figures/supplement_analysis.pdf'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
