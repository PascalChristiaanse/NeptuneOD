import numpy as np
import spiceypy as spice

# ==========================================
# 1. LOAD THE HIGH-PRECISION KERNELS
# ==========================================
# Make sure you have downloaded these from NAIF / ESA Gaia repositories
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/naif0012.tls")       # Leapseconds kernel (LSK)
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/DE440.bsp")          # High-precision planetary ephemeris
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/nep097.bsp")         # basic solar system bodies ephemeris
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/F8B24.bsp")         # High-precision solar system bodies
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/F8M24.bsp")         # High-precision Neptune/Triton ephemeris
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/gaia_v01.tf")        # Gaia frames kernel (defines GAIA ID)
spice.furnsh("/home/pascal/Documents/NeptuneOD/data/kernels/gaia_flp.bsp") # Real/reconstructed Gaia trajectory SPK


# ==========================================
# 2. CONVERT GAIA TCB TO SPICE TDB
# ==========================================
def gaia_tcb_to_tdb_et(tcb_jd):
    """
    Converts a Gaia Julian Date in TCB (Barycentric Coordinate Time)
    to SPICE Ephemeris Time (TDB seconds past J2000).
    Uses the IAU 2006 linear relationship.
    """
    # TDB seconds past J2000 for the JD epoch
    tdb_jd_j2000 = 2451545.0
    
    # Standard IAU linear drift rate between TCB and TDB
    # L_B = 1.550519768 * 10^-8
    LB = 1.550519768e-8
    
    # Convert input TCB JD into TCB seconds past J2000
    tcb_sec = (tcb_jd - tdb_jd_j2000) * 86400.0
    
    # TDB = TCB - L_B * (TCB_seconds) - TDB_0
    # At J2000.0, TCB - TDB is approximately -65.56 ms
    tdb_0 = -6.5562e-2 
    
    et_tdb = tcb_sec - (LB * tcb_sec) - tdb_0
    return et_tdb


# Example Gaia observation time (Julian Date in TCB)
gaia_epoch_tbc_days_since_j2010 = 1761.0780768803459
J2010_jd = 2455197.5
gaia_tcb_epoch_jd = J2010_jd + gaia_epoch_tbc_days_since_j2010
et = gaia_tcb_to_tdb_et(gaia_tcb_epoch_jd)


# ==========================================
# 3. SPICE QUERY FOR TRITON FROM GAIA
# ==========================================
# Target: TRITON (ID 801)
# Observer: GAIA (ID -123 or 'GAIA')
# Reference Frame: J2000 (Maps natively to ICRF in DE440)
# Aberration Correction: 'XCN+S' (Converged Newtonian light time + stellar aberration)
#                        The 'X' prefix enables relativistic transmission corrections.

position, light_time = spice.spkpos(
    targ="TRITON",
    et=et,
    ref="J2000",
    abcorr="CN",
    # abcorr="XCN+S",
    obs="GAIA"
)


# ==========================================
# 4. CONVERT POSITION VECTOR TO RA/DEC
# ==========================================
# Convert the rectangular position vector (km) to Right Ascension and Declination
range_km, ra_rad, dec_rad = spice.recrad(position)

# Convert radians to degrees
ra_deg = ra_rad * spice.dpr()
dec_deg = dec_rad * spice.dpr()

print(f"Gaia TCB JD: {gaia_tcb_epoch_jd}")
print(f"Calculated RA (deg):  {ra_deg:.12f}")
print(f"Calculated Dec (deg): {dec_deg:.12f}")
print(f"Light time (seconds): {light_time:.6f}")


# ==========================================
# 5. GAIA ARCHIVE REFERENCE VALUES
# ==========================================
# Reference astrometry from the Gaia archive (ICRS, corrected for full
# relativistic aberration, but NOT for relativistic light deflection in the
# Solar System).
ra_from_gaia_archive = 336.8083943124863
dec_from_gaia_archive = -10.490687886968159

# Position angle of the scan direction at the epoch of observation (deg).
# From the Gaia archive column `position_angle_scan`:
#   0 = North, 90 = increasing RA, 180 = South, 270 = decreasing RA.
# It is the angle between the along-scan (AL) direction and the direction to
# the North Pole at the SSO position.
position_angle_scan_deg = 303.0407249788776   # <-- SET THIS from the Gaia archive

print(f"RA from Gaia archive (deg): {ra_from_gaia_archive:.12f}")
print(f"Dec from Gaia archive (deg): {dec_from_gaia_archive:.12f}")
print(f"Position angle of scan (deg): {position_angle_scan_deg:.6f}")


# ==========================================
# 6. ALONG-SCAN / ACROSS-SCAN RESIDUALS
# ==========================================
# Residuals in RA and Dec (radians)
d_ra = (ra_deg - ra_from_gaia_archive) * np.pi / 180.0
d_dec = (dec_deg - dec_from_gaia_archive) * np.pi / 180.0

# Physical angular offset in the tangent plane (radians):
#   east  = d_ra * cos(dec)   (direction of increasing RA)
#   north = d_dec             (direction of increasing Dec)
dec_rad_archive = dec_from_gaia_archive * np.pi / 180.0
east = d_ra * np.cos(dec_rad_archive)
north = d_dec

# Scan angle (radians), measured from North towards East
psi = position_angle_scan_deg * np.pi / 180.0

# Along-scan (AL) direction: unit vector in (east, north) coordinates
#   AL = (sin psi, cos psi)
# Across-scan (AC) direction: perpendicular to AL
#   AC = (cos psi, -sin psi)
al_resid = east * np.sin(psi) + north * np.cos(psi)
ac_resid = east * np.cos(psi) - north * np.sin(psi)

# Convert to arcseconds
arcsec_per_rad = 180.0 / np.pi * 3600.0
al_resid_arcsec = al_resid * arcsec_per_rad
ac_resid_arcsec = ac_resid * arcsec_per_rad

print(f"RA difference (arcsec): {d_ra * arcsec_per_rad:.6f}")
print(f"Dec difference (arcsec): {d_dec * arcsec_per_rad:.6f}")
print(f"Along-scan residual (arcsec): {al_resid_arcsec:.6f}")
print(f"Across-scan residual (arcsec): {ac_resid_arcsec:.6f}")

# Total angular error (arcsec)
err_rad = np.sqrt(east**2 + north**2)
print(f"Total angular error (arcsec): {err_rad * arcsec_per_rad:.6f}")

# Range * angular error
max_km_error = np.linalg.norm(position) * err_rad
print(f"Max position error due to angular error (km): {max_km_error:.6f}")