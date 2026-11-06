"""Opening figure of the talk: 8 Gaussians -> two moons with the OT-CFM model from Flow_matching_tutorial.ipynb.

    python slides/make_opening_figs.py slides/figs

Outputs
    opening_2d.gif   noise -> data with the paths drawn as they grow (online deck)
    opening_2d.png   last frame of the same animation (used in the PDF instead of the GIF)
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "slides" / "figs")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchcfm.models.models import MLP
from torchcfm.utils import sample_8gaussians

BG, GRAY, ORANGE, NAVY = "#FAF8F3", "#8A94A3", "#D9822B", "#14213D"
STEPS, N_POINTS, N_PATHS = 60, 1024, 300
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

net = MLP(dim=2, time_varying=True)
net.load_state_dict(torch.load(ROOT / "conditional-flow-matching-main/examples/2D_tutorials/moon_optimal.pt", map_location="cpu"))
net.eval()

torch.manual_seed(0)
x = sample_8gaussians(N_POINTS)
traj = [x.numpy()]
with torch.no_grad():
    for k in range(STEPS):
        t = torch.full((N_POINTS, 1), k / STEPS)
        x = x + net(torch.cat([x, t], dim=-1)) / STEPS
        traj.append(x.numpy())
traj = np.stack(traj)


def frame(k):
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=100)
    ax.plot(traj[: k + 1, :N_PATHS, 0], traj[: k + 1, :N_PATHS, 1], c=ORANGE, lw=0.5, alpha=0.18)
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=5, c=GRAY, alpha=0.6)
    ax.scatter(traj[k, :, 0], traj[k, :, 1], s=7, c=ORANGE)
    ax.text(0.02, 0.97, f"t = {k / STEPS:.2f}", transform=ax.transAxes, va="top", color=NAVY, fontsize=17)
    ax.set_xlim(-8, 8); ax.set_ylim(-7, 7); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


frames = [frame(k) for k in range(0, STEPS + 1)]
frames[-1].save(OUT / "opening_2d.png")
frames = [f.quantize(colors=48, dither=Image.Dither.NONE) for f in frames]
frames[0].save(OUT / "opening_2d.gif", save_all=True, append_images=frames[1:] + [frames[-1]] * 25,
               duration=70, loop=0, optimize=True)
print("saved", OUT / "opening_2d.gif")
