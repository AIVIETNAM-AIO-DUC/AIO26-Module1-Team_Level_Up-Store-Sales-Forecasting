"""Generate the illustrative Fourier/periodogram teaching figures for the Stage 0 docs.

These are *illustrative* (synthetic), not data analysis: they explain how a periodogram
turns a curve into spikes. The spikes are produced by a real ``np.fft.rfft`` on the
synthetic signals -- nothing is hand-drawn -- so the picture is honest about what the
FFT actually reports.

Output: ``docs/assets/fourier-smooth-vs-sharp.png`` (referenced from
``docs/en/stage-0/`` and ``docs/vi/stage-0/`` periodogram sections).

Run:  uv run python docs/assets/make_fourier_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "fourier-smooth-vs-sharp.png"

# Store 1, GROCERY I mean sales by weekday (Mon..Sun) -- the SAME numbers as the
# weekly bar chart already on the Stage 0 page. Flat Mon-Sat, a sharp Sunday dip.
WEEKDAY = np.array([2383, 2409, 2770, 2229, 2414, 2323, 1031], dtype=float)

WEEKS = 12                      # repeats: enough for clean frequency resolution
N = WEEKS * 7                   # daily samples
t = np.arange(N)


def periodogram(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (period_in_days, power) for the 2-14 day band, mean removed."""
    x = x - x.mean()                       # drop the constant level
    freqs = np.fft.rfftfreq(N, d=1.0)      # cycles per day
    power = np.abs(np.fft.rfft(x)) ** 2
    periods = np.full_like(freqs, np.inf)
    periods[1:] = 1.0 / freqs[1:]          # days per cycle (skip freq 0)
    band = (periods >= 2) & (periods <= 14)
    return periods[band], power[band]


# --- two signals with the SAME 7-day period but different SHAPE -----------------
amp = (WEEKDAY.max() - WEEKDAY.min()) / 2
smooth = WEEKDAY.mean() + amp * np.cos(2 * np.pi * t / 7)   # one pure 7-day cosine
sharp = np.tile(WEEKDAY, WEEKS)                             # real flat-week + Sun dip

fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)

for row, (sig, label, color, note) in enumerate(
    [
        (smooth, "Smooth 7-day cosine", "C0", "one clean spike at 7 days"),
        (sharp, "Sharp weekly shape (store 1)", "C2", "7-day spike + 3.5-day harmonic"),
    ]
):
    # time domain (show 4 weeks so the shape is legible)
    ax = axes[row, 0]
    ax.plot(t[: 4 * 7], sig[: 4 * 7], color=color, lw=1.8)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("day")
    ax.set_ylabel("sales")
    for w in range(0, 4 * 7 + 1, 7):
        ax.axvline(w, color="0.85", lw=0.8, zorder=0)

    # frequency domain
    per, pw = periodogram(sig)
    ax = axes[row, 1]
    ax.semilogy(per, pw, color=color, lw=1.6)
    ax.set_ylim(pw.max() * 1e-6, pw.max() * 4)   # keep numerical floor out of view
    for p in (7, 3.5):
        ax.axvline(p, color="C3", ls="--", lw=1)
    ax.set_title("periodogram (FFT power)", fontsize=10)
    ax.set_xlabel("cycle length (days)")
    ax.set_ylabel("power (log)")
    ax.set_xticks([2, 3.5, 7, 10, 14])
    ax.text(0.97, 0.92, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="0.25",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))

fig.suptitle("Same 7-day period, different shape — a sharp cycle needs a 3.5-day harmonic",
             fontsize=12)
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")

# Report the spikes so the figure's claim is verifiable from the console.
per, pw = periodogram(sharp)
top = np.argsort(pw)[::-1][:3]
print("sharp-shape strongest cycles (days):", ", ".join(f"{per[i]:.1f}" for i in top))
