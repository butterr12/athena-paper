import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})

BASE = '/home/alfonsoluisdelosreyes/school/198-thesis/thesis/Template__CS_199_Research_Manuscripts/figures/'

# ── Graph 1: Backbone comparison (best config per backbone, Li et al. dataset) ──
fig, ax = plt.subplots(figsize=(8, 4.5))

backbones  = ['SmolVLM\n(LoRA32, DoRA=Y)', 'BLIP-2\n(LoRA32, DoRA=Y)', 'Qwen3.5\n(LoRA8, DoRA=Y)']
bleu1  = [0.7820, 0.6639, 0.7877]
bleu4  = [0.4332, 0.3335, 0.4568]
rougel = [0.6574, 0.5322, 0.6765]
cider  = [1.2810, 0.8740, 1.4464]
spice  = [0.2610, 0.2709, 0.2918]
meteor = [0.3241, 0.3335, 0.3572]

x = np.arange(len(backbones))
w = 0.13
colors = ['#2A9D8F', '#F4A261', '#E63946', '#457B9D', '#6D6875', '#E9C46A']
labels_ = ['BLEU-1', 'BLEU-4', 'ROUGE-L', 'CIDEr (÷2)', 'SPICE', 'METEOR']
data = [bleu1, bleu4, rougel, [c/2 for c in cider], spice, meteor]

for i, (vals, col, lab) in enumerate(zip(data, colors, labels_)):
    bars = ax.bar(x + (i - 2.5) * w, vals, width=w, color=col, label=lab,
                  edgecolor='white', linewidth=0.5, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(backbones, fontsize=9.5)
ax.set_ylabel('Score', fontsize=11)
ax.set_ylim(0, 0.97)
ax.set_title('Best Configuration per Backbone (Li et al. Dataset)', fontsize=11, fontweight='bold', pad=10)
ax.legend(fontsize=9, framealpha=0.35, ncol=6, loc='upper left')
ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)
# Highlight Qwen3.5 group and label as selected
ax.axvspan(1.5, 2.5, alpha=0.07, color='#E63946', zorder=0)
ax.text(2, 0.845, 'Selected', ha='center', fontsize=9, color='#E63946',
        fontweight='bold')
ax.annotate('', xy=(2, 0.808), xytext=(2, 0.838),
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.2))

fig.tight_layout()
fig.savefig(BASE + 'fig_backbone_comparison.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_backbone_comparison.pdf')
plt.close()

# ── Graph 2: VRAM vs Performance bubble chart ──
fig, ax = plt.subplots(figsize=(7, 5))

models_ = ['SmolVLM\n(fp16)', 'BLIP-2\n(fp16)', 'Qwen3.5\n(fp16)',
           'SmolVLM\n(4-bit)', 'BLIP-2\n(4-bit)', 'Qwen3.5\n(4-bit)']
vram_   = [1490, 2200, 1770, 604, 966, 653]
score_  = [0.7820, 0.6639, 0.7862, 0.7820, 0.6639, 0.7862]
colors_ = ['#B0C4DE', '#D2B48C', '#FFB6C1', '#2A9D8F', '#F4A261', '#E63946']
sizes_  = [200, 200, 200, 200, 200, 200]

for (m, v, s, c) in zip(models_, vram_, score_, colors_):
    ax.scatter(v, s, s=280, color=c, edgecolors='white', linewidths=1.5, zorder=3)
    ax.annotate(m, xy=(v, s), xytext=(v + 30, s + 0.005), fontsize=8, ha='left', va='bottom')

ax.axvline(x=8000, color='grey', linestyle='--', linewidth=1, alpha=0.5, label='Jetson Orin 8 GB limit')
ax.set_xlabel('VRAM Usage (MB)', fontsize=11)
ax.set_ylabel('Best BLEU-1', fontsize=11)
ax.set_title('VRAM Footprint vs. Captioning Performance\n(fp16 and 4-bit NF4)', fontsize=11, fontweight='bold', pad=10)
ax.set_xlim(400, 3000)
ax.set_ylim(0.60, 0.83)
ax.legend(fontsize=9, framealpha=0.35)
ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)

# Add legend patches for fp16 vs 4-bit
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='#D3D3D3', label='fp16'), Patch(facecolor='#2A9D8F', label='4-bit NF4 (deployment)')]
ax.legend(handles=legend_els + [plt.Line2D([0],[0], color='grey', linestyle='--', label='Jetson Orin 8 GB')],
          fontsize=9, framealpha=0.35, loc='lower right')

fig.tight_layout()
fig.savefig(BASE + 'fig_vram_vs_perf.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_vram_vs_perf.pdf')
plt.close()

# ── Graph 3: DoRA vs no-DoRA effect across ranks (Qwen3.5, Li et al., Prep=OFF) ──
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
axes = axes.flatten()

ranks = [8, 16, 32]
bleu1_nodora  = [0.7809, 0.7715, 0.7749]
bleu1_dora    = [0.7877, 0.7778, 0.7813]
cider_nodora  = [1.4234, 1.4340, 1.3648]
cider_dora    = [1.4464, 1.4420, 1.3705]
spice_nodora  = [0.2790, 0.2830, 0.2744]
spice_dora    = [0.2918, 0.2868, 0.2764]
meteor_nodora = [0.3485, 0.3502, 0.3363]
meteor_dora   = [0.3572, 0.3476, 0.3381]

panels = [
    (bleu1_nodora,  bleu1_dora,  'BLEU-1',  'BLEU-1: DoRA vs. No DoRA'),
    (cider_nodora,  cider_dora,  'CIDEr',   'CIDEr: DoRA vs. No DoRA'),
    (spice_nodora,  spice_dora,  'SPICE',   'SPICE: DoRA vs. No DoRA'),
    (meteor_nodora, meteor_dora, 'METEOR',  'METEOR: DoRA vs. No DoRA'),
]

for ax_, (no, yes, ylabel, title) in zip(axes, panels):
    ax_.plot(ranks, no,  'o--', color='#457B9D', lw=2, ms=8, label='DoRA=False')
    ax_.plot(ranks, yes, 's-',  color='#E63946', lw=2, ms=8, label='DoRA=True')
    ax_.fill_between(ranks, no, yes, alpha=0.10, color='#E63946')
    ax_.set_xlabel('LoRA Rank', fontsize=10)
    ax_.set_ylabel(ylabel, fontsize=10)
    ax_.set_xticks(ranks)
    ax_.set_title(title + '\n(Qwen3.5, Li et al. Dataset, Prep=OFF)', fontsize=9.5, fontweight='bold')
    ax_.legend(fontsize=9, framealpha=0.35)
    ax_.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax_.set_axisbelow(True)

fig.tight_layout()
fig.savefig(BASE + 'fig_dora_effect.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved fig_dora_effect.pdf')
plt.close()
