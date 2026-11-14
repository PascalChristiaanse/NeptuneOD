import numpy as np
from scipy.constants import arcsec
from scipy.linalg import block_diag

# One transit: 8 observations (obs 11..18). Columns:
# ra_err_sys, dec_err_sys, corr_sys, ra_err_rand, dec_err_rand, corr_rand  (all mas)
rows = [
    (
        1.2547690685417077,
        1.9282519505062028,
        0.9988241111363347,
        333.9289102768165,
        513.406069206838,
        0.9999996218595296,
    ),
    (
        1.2547739191468579,
        1.9282487940690731,
        0.9988241163810136,
        333.93020088515215,
        513.4052275400686,
        0.999999636463222,
    ),
    (
        1.2547789461180818,
        1.9282455228468172,
        0.9988241218162962,
        333.93154019262033,
        513.4043557536282,
        0.9999996408450441,
    ),
    (
        1.2547839786155244,
        1.928242248009776,
        0.998824127257461,
        333.9328806142833,
        513.4034829052919,
        0.9999996474034554,
    ),
    (
        1.2547890156732084,
        1.9282389701864908,
        0.9988241327034637,
        333.9342234644604,
        513.4026095950483,
        0.9999996466063464,
    ),
    (
        1.254794056350247,
        1.9282356899891915,
        0.998824138153287,
        333.9355678155945,
        513.4017357998108,
        0.9999996425577268,
    ),
    (
        1.2547990995553755,
        1.9282324081279776,
        0.9988241436057504,
        333.93691139143556,
        513.4008611624488,
        0.99999964729761,
    ),
    (
        1.2548039780308629,
        1.9282292334465632,
        0.9988241488800278,
        333.9382123219853,
        513.4000154306617,
        0.9999996443576321,
    ),
]
dec_deg = -10.490687886968159
dec = np.deg2rad(dec_deg)

mas_to_rad = arcsec / 1e3
arr = np.array(rows)
arr[:, [0, 1, 3, 4]] *= mas_to_rad  # mas -> rad
arr[:, [0, 3]] /= np.cos(dec)  # remove cos(dec) from RA errors


def comp(ra, dec_, corr):
    return np.array([[ra**2, ra * dec_ * corr], [ra * dec_ * corr, dec_**2]])


# random block-diagonal
C_rand = block_diag(*[comp(ra, de, co) for ra, de, co in arr[:, 3:6]])
# systematic from first row, tiled
C_sys_sub = comp(*arr[0, 0:3])
n = len(rows)
C_sys = np.tile(C_sys_sub, (n, n))
C = C_rand + C_sys

W = np.linalg.inv(C)
W = (W + W.T) / 2

np.set_printoptions(precision=6, suppress=False, linewidth=200)
print("cos(dec) =", np.cos(dec))
print("\nPer-observation random 2x2 (obs 1):\n", comp(*arr[0, 3:6]))
print("\nSystematic 2x2 submatrix:\n", C_sys_sub)
print("\nFull 16x16 covariance C:\n", C)
print("\nFull 16x16 weight W = inv(C) symmetrized:\n", W)
print("\nDiagonal of W (per-observation weights):\n", np.diag(W))
print("\nW[0:2,0:2] (obs1 block):\n", W[0:2, 0:2])
print("\nW[0:2,2:4] (obs1-obs2 coupling):\n", W[0:2, 2:4])
