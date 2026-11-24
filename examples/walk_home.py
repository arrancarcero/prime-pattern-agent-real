"""Pictures behind the walk skill.

    python examples/walk_home.py        -> walk_home.png

Top: the one-way (eta) walk at the first zero spirals in to the origin; between zeros it
     settles somewhere else.
Bottom: the two-way (Riemann-Siegel) walk at the 100th zero: M arrows out, then the mirror
     walk comes back "from infinity", one small correction arrow (the Riemann-Siegel
     remainder) closes the gap, and it lands at the origin; between zeros it misses.
"""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppa_real.skills.candles import ZEROS  # noqa: E402
from ppa_real.skills.spacing import hardy_z  # noqa: E402
from ppa_real.skills.walk import eta_averaged, eta_stops, two_way  # noqa: E402

Z100 = 236.5242296658                       # 100th zeta zero (mpmath)
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

for col, (t, label) in enumerate([(ZEROS[0], f"at the first zero, t = {ZEROS[0]:.3f}"),
                                  ((ZEROS[0] + ZEROS[1]) / 2, f"between zeros, t = {(ZEROS[0] + ZEROS[1]) / 2:.3f}")]):
    S = np.concatenate([[0], eta_stops(t, 3000)])
    A = eta_averaged(S)
    a = ax[0, col]
    a.plot(S.real, S.imag, color="#90cdf4", lw=0.7, label="arrow by arrow (3,000 arrows)")
    a.plot(A.real, A.imag, color="#2b6cb0", lw=1.5, label="midpoints (averaged)")
    a.plot(0, 0, "o", color="#c05621", ms=9, label="origin (home)")
    end = eta_averaged(eta_stops(t, 400_000))[-1]
    a.plot(end.real, end.imag, "*", color="k", ms=14, label=f"where it settles: |end| = {abs(end):.2g}")
    a.set_title(f"One-way walk {label}")
    a.set_aspect("equal")
    a.legend(fontsize=8, loc="upper right")

for col, (t, label) in enumerate([(Z100, f"at the 100th zero, t = {Z100:.3f}"),
                                  (Z100 + 0.45, f"off the zero, t = {Z100 + 0.45:.3f}")]):
    M, path = two_way(t)
    a = ax[1, col]
    a.plot(path[:M + 1].real, path[:M + 1].imag, "-o", color="#2b6cb0", ms=4, label=f"{M} arrows forward")
    a.plot(path[M:].real, path[M:].imag, "-o", color="#c05621", ms=4, label=f"{M} mirror arrows back from infinity")
    z = hardy_z(t)[0]
    a.annotate("", xy=(z, 0), xytext=(path[-1].real, path[-1].imag),
               arrowprops=dict(arrowstyle="->", color="#2f855a", lw=2.5))
    a.plot([], [], color="#2f855a", lw=2.5, label=f"1 correction arrow: {path[-1].real:+.3f} -> {z:+.3f}")
    a.plot(0, 0, "o", color="k", ms=11, mfc="none", label="origin (home)")
    a.plot(z, 0, "*", color="k", ms=12, label=f"lands at Z = {z:+.3f}")
    a.set_title(f"Two-way walk {label}")
    a.set_aspect("equal")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

fig.tight_layout()
fig.savefig("walk_home.png", dpi=120)
