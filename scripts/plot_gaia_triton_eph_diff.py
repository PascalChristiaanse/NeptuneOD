#!/usr/bin/env python3
"""Plot the Triton orbit difference between F8M24 and NEP097.

Important: the paper-style comparison is **Triton relative to each ephemeris's
OWN Neptune system barycenter (NAIF 899)**, with DE440 loaded only as the SSB
anchor (to chain 801/899 -> SSB).  A naively common reference (e.g. subtracting
DE440's Neptune barycenter NAIF 8 from both) cancels the real NSB(899)
difference and gives ~0 km, which is wrong.

Each candidate SPK is loaded in a *fresh* kernel pool together with DE440, so
no kernel shadows another and DE440 provides the SSB chain.  The difference
between two candidates is (801-899)_A - (801-899)_B, decomposed into radial
(R), transverse (T), and out-of-plane (N) components, matching the paper.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spiceypy as spy

PROJECT = Path(__file__).resolve().parent.parent
K = PROJECT / "data" / "kernels"
TABLE = PROJECT / "data" / "gaia" / "triton.pkl"

BASE_KERNELS = [str(K / "naif0012.tls"), str(K / "pck00010.tpc")]
DE440 = str(K / "DE440.bsp")

FRAME = "J2000"
ID_TRITON = 801
ID_NSB = 899
ID_SSB = 0

# Reference = F8M24 (newly fitted ephemeris).
REF_NAME = "F8M24"
REF_FILE = str(K / "F8M24.bsp")
CANDIDATES = [("NEP097", str(K / "nep097.bsp"))]

NEPTUNE_DIST_AU = 30.1
AU = 149597870.7


def load_fresh(extra_kernels):
    """Reset the kernel pool and load base + DE440 + extra kernels fresh."""
    spy.kclear()
    for kf in BASE_KERNELS + [DE440] + list(extra_kernels):
        spy.furnsh(kf)


def triton_rel_nsb(ets, triton_file):
    """Triton(801) rel NSB(899), both from *triton_file* (km)."""
    load_fresh([triton_file])
    out = []
    for et in ets:
        t_st, _ = spy.spkgeo(ID_TRITON, float(et), FRAME, ID_SSB)
        n_st, _ = spy.spkgeo(ID_NSB, float(et), FRAME, ID_SSB)
        out.append(t_st[:3] - n_st[:3])
    return np.array(out)


def seconds_since_j2000(year: float) -> float:
    """Approximate seconds since J2000 TDB for a calendar year (1 Jan)."""
    # 2000-01-01T12:00 = 0; one Julian year = 365.25 days
    return (year - 2000.0) * 365.25 * 86400.0


def main():
    table = pd.read_pickle(str(TABLE))
    epochs = table["epoch"].to_numpy().astype(float)

    # Plot over the requested long arc (1900-2050), not just the Gaia window.
    year_min, year_max = 1900.0, 2050.0
    et_dense = np.linspace(
        seconds_since_j2000(year_min), seconds_since_j2000(year_max), 2000
    )
    years = et_dense / 31556952.0 + 2000.0  # approx calendar years for plotting

    ref = triton_rel_nsb(et_dense, REF_FILE)

    # R/T/N basis from the mean reference orbit and its velocity.
    vel = []
    load_fresh([REF_FILE])
    for et in et_dense:
        s801, _ = spy.spkgeo(ID_TRITON, float(et), FRAME, ID_SSB)
        s899, _ = spy.spkgeo(ID_NSB, float(et), FRAME, ID_SSB)
        vel.append(s801[3:6] - s899[3:6])
    vel = np.array(vel)

    rhat = ref.mean(axis=0)
    rhat /= np.linalg.norm(rhat)
    vhat = vel.mean(axis=0)
    vhat /= np.linalg.norm(vhat)
    nhat = np.cross(rhat, vhat)
    nhat /= np.linalg.norm(nhat)
    that = np.cross(nhat, rhat)

    dist_km = NEPTUNE_DIST_AU * AU

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name, f in CANDIDATES:
        pos = triton_rel_nsb(et_dense, f)
        dr = pos - ref
        R = dr @ rhat
        T = dr @ that
        N = dr @ nhat
        d3 = np.linalg.norm(dr, axis=1)
        axes[0, 0].plot(years, R, label=name)
        axes[0, 1].plot(years, T, label=name)
        axes[1, 0].plot(years, N, label=name)
        axes[1, 1].plot(years, d3, label=name)

    titles = ["Radial R [km]", "Transverse T [km]", "Out-of-plane N [km]", "3D norm |Δd| [km]"]
    for a, t in zip(axes.flat, titles):
        a.set_title(t)
        a.set_xlabel("Year")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle(f"Triton rel NSB: {REF_NAME} vs NEP097 (own NSB each)")
    fig.tight_layout()

    out = PROJECT / "scripts" / "triton_eph_difference.pdf"
    fig.savefig(str(out))
    fig.savefig(str(out.with_suffix(".png")))

    # Summary over the full plotted arc (1900-2050).
    print(f"{'candidate':24s} {'mean km':>8s} {'max km':>8s} {'max arcsec':>10s}")
    for name, f in CANDIDATES:
        pos = triton_rel_nsb(et_dense, f)
        dr = pos - ref
        d3 = np.linalg.norm(dr, axis=1)
        R = dr @ rhat
        T = dr @ that
        N = dr @ nhat
        ang = np.degrees(np.arcsin(np.clip(d3 / dist_km, 0, 1))) * 3600.0
        print(f"{name:24s} {d3.mean():8.0f} {d3.max():8.0f} {ang.max():10.2f}")
        print(f"    R/T/N rms: {np.sqrt(np.mean(R**2)):.0f} / "
              f"{np.sqrt(np.mean(T**2)):.0f} / {np.sqrt(np.mean(N**2)):.0f} km")
        print(f"    at Gaia window (2014-2019): mean {np.mean(d3[(years>=2014)&(years<=2019)]):.0f} "
              f"max {np.max(d3[(years>=2014)&(years<=2019)]):.0f} km")
    print(f"{REF_NAME:24s}  (reference, zero by construction)")
    print("saved", out)


if __name__ == "__main__":
    main()