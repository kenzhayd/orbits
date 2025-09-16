"""
Utilities for organizing and fitting multi-star astrometric data with Octofitter.
Includes:
- StarData class for storing motion and error data
- stars dictionary with input for each star
- Functions to build models, propagate errors, and run Octofitter fits and plots
"""

#Environment variables
import os
os.environ["JULIA_NUM_THREADS"] = "auto"
os.environ["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"


# Imports
import numpy as np
import astropy.units as u
import astropy.constants as const
import pandas as pd
from astropy.table import Table
from datetime import datetime
from scipy import stats

import octofitterpy as octo


# ========== Define StarData Class ==========
class StarData:
    def __init__(self, name, ra, dec, pm_ra, pm_dec, acc_ra, acc_dec,
                 sigma_pm_ra, sigma_pm_dec, sigma_acc_ra, sigma_acc_dec, v2D, v2D_err):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.pm_ra = pm_ra
        self.pm_dec = pm_dec
        self.acc_ra = acc_ra
        self.acc_dec = acc_dec
        self.sigma_pm_ra = sigma_pm_ra
        self.sigma_pm_dec = sigma_pm_dec
        self.sigma_acc_ra = sigma_acc_ra
        self.sigma_acc_dec = sigma_acc_dec
        self.v2D = v2D
        self.v2D_err = v2D_err

# ========== Omega Centauri Data ==========
# We find the center to be at (α,δ) = (13:26:47.24, −47:28:46.45).
# from https://iopscience.iop.org/article/10.1088/0004-637X/710/2/1032
# reference 6 in Haberle et al. (2024; Nature, Vol. 631) 
# We assume the IMBH is located at the cluster center
ra_cm_mas  = (201.696834 * u.deg).to(u.mas).value
dec_cm_mas = (-47.479569 * u.deg).to(u.mas).value

# Distance to Omega Centauri center (km)
distance_kpc = 5.43 * u.kpc
distance_km= distance_kpc.to(u.km)

# Assumed errors in position
ra_err = 0.5*u.mas
dec_err = 0.5*u.mas

# ========== Define Error Propagation Function ==========
def propagate_error(sigma_pos, sigma_pm, sigma_acc, dt):
    """
    Propagates uncertainty in position due to uncertainties in
    proper motion and acceleration over time. No uncertainty in initial position is available.

    Parameters:
    - sigma_pos : Initial uncertainty in position (mas)
    - sigma_pm  : Uncertainty in proper motion (mas/yr)
    - sigma_acc : Uncertainty in acceleration (mas/yr²)
    - dt        : Time from reference epoch (in years)

    Returns:
    - sigma_pos : Total uncertainty in predicted position at time dt (mas)
    """
    term_pos = sigma_pos**2                      # variance from initial position
    term_pm = (dt * sigma_pm) ** 2               # variance from proper motion
    term_acc = (0.5 * dt**2 * sigma_acc) ** 2    # variance from acceleration

    sigma_pos_total = np.sqrt(term_pos + term_pm + term_acc)
    return sigma_pos_total

# ========== Define Position Projection Function ==========
def fake_pos(pos, pm, acc, dt):
    """
    Gives a fake observed position using an observed angular position, velocity, and acceleration

    Parameters:
    - pos : Initial position (in mas)
    - pm : Proper motion (in mas/yr)
    - acc : Acceleration (in mas/yr²)
    - dt : Time offset(s) from the reference epoch in years

    Returns:
    - pos_final : calculated position (in mas)
    """
    return pos + pm * dt + 0.5 * acc * dt**2

# ========== Define Star Dictionary ==========
stars = {
    "A": StarData(
        name="A",
        ra=(201.6967263 * u.deg).to(u.mas).value, # mas
        dec=(-47.4795835 * u.deg).to(u.mas).value, # mas
        pm_ra=3.563, # mas/year
        pm_dec=2.564, # mas/year
        acc_ra=-0.0069, # mas/year^2
        acc_dec=0.0085, # mas/year^2
        sigma_pm_ra=0.038, # mas/year
        sigma_pm_dec=0.055, # mas/year
        sigma_acc_ra=0.0083, # mas/year^2
        sigma_acc_dec=0.0098, # mas/year^2
        v2D=113.0,         # km/s
        v2D_err=1.1        # km/s
    ),
    "B": StarData(
        name="B",
        ra=(201.6968888 * u.deg).to(u.mas).value, # mas
        dec=(-47.4797138 * u.deg).to(u.mas).value, # mas
        pm_ra=2.167, # mas/year
        pm_dec=1.415, # mas/year
        acc_ra=0.0702, # mas/year^2
        acc_dec=0.0228, # mas/year^2
        sigma_pm_ra=0.182, # mas/year
        sigma_pm_dec=0.081, # mas/year
        sigma_acc_ra=0.0239, # mas/year^2
        sigma_acc_dec=0.0157, # mas/year^2
        v2D=66.6,
        v2D_err=4.1
    ),

    "C": StarData(
        name="C",
        ra=(201.6966378 * u.deg).to(u.mas).value, # mas
        dec=(-47.4793672 * u.deg).to(u.mas).value, # mas
        pm_ra=1.117, # mas/year
        pm_dec=3.514, # mas/year
        acc_ra=0.0028, # mas/year^2
        acc_dec=-0.0060, # mas/year^2
        sigma_pm_ra=0.127 , # mas/year
        sigma_pm_dec=0.056 , # mas/year
        sigma_acc_ra=0.0333, # mas/year^2
        sigma_acc_dec=0.0123, # mas/year^2
        v2D=94.9,
        v2D_err=1.7
    ),

    "D": StarData(
        name="D",
        ra=(201.6968346 * u.deg).to(u.mas).value, # mas
        dec=(-47.4793233 * u.deg).to(u.mas).value, # mas
        pm_ra=2.559, # mas/year
        pm_dec=-1.617, # mas/year
        acc_ra=0.0357, # mas/year^2
        acc_dec=-0.0194, # mas/year^2
        sigma_pm_ra=0.082, # mas/year
        sigma_pm_dec=0.061 , # mas/year
        sigma_acc_ra=0.0177, # mas/year^2
        sigma_acc_dec=0.0162, # mas/year^2
        v2D=77.9,
        v2D_err=2.0
    ),

    "E": StarData(
        name="E",
        ra=(201.6973080 * u.deg).to(u.mas).value, # mas
        dec=(-47.4797545 * u.deg).to(u.mas).value, # mas
        pm_ra=-2.149, # mas/year
        pm_dec=1.638, # mas/year
        acc_ra=0.0072, # mas/year^2
        acc_dec=-0.0009, # mas/year^2
        sigma_pm_ra=0.025, # mas/year
        sigma_pm_dec=0.037, # mas/year
        sigma_acc_ra=0.0042, # mas/year^2
        sigma_acc_dec=0.0075, # mas/year^2
        v2D=69.6,
        v2D_err=0.8
    ),

    "F": StarData(
        name="F",
        ra=(201.6977125 * u.deg).to(u.mas).value, # mas
        dec=(-47.4792625 * u.deg).to(u.mas).value, # mas
        pm_ra=0.436, # mas/year
        pm_dec=-2.584, # mas/year
        acc_ra=0.0052, # mas/year^2
        acc_dec=-0.0015, # mas/year^2
        sigma_pm_ra=0.017, # mas/year
        sigma_pm_dec=0.016, # mas/year
        sigma_acc_ra=0.0038, # mas/year^2
        sigma_acc_dec=0.0038, # mas/year^2
        v2D=67.4,
        v2D_err=0.4
    ),

    "G": StarData(
        name="G",
        ra=(201.6961340 * u.deg).to(u.mas).value, # mas
        dec=(-47.4790585 * u.deg).to(u.mas).value, # mas
        pm_ra=-1.317, # mas/year
        pm_dec=2.207, # mas/year
        acc_ra=-0.0197, # mas/year^2
        acc_dec=0.0173, # mas/year^2
        sigma_pm_ra=0.098, # mas/year
        sigma_pm_dec=0.062, # mas/year
        sigma_acc_ra=0.0267, # mas/year^2
        sigma_acc_dec=0.0170, # mas/year^2
        v2D=66.2,
        v2D_err=1.9
    ),

        }
# ========== Astrometry Imput for Octofitter ==========
def simulate_astrometry(star, epoch, dt):
    """
    Simulates astrometric positions for a star at three epochs: past, present, and future.
    
    Parameters:
    - star: StarData object containing motion and error data
    - epoch: Central epoch in calendar years (e.g., 2010)
    - dt: Time offset from the central epoch in years (e.g., 10)

    Returns:
    - ra_rel: Relative right ascension values at the three epochs (mas)
    - dec_rel: Relative declination values at the three epochs (mas)
    - ra_errs: Measurement errors for RA at the three epochs (mas)
    - dec_errs: Measurement errors for Dec at the three epochs (mas)
    """
    
    # --- Time setup: generate observation epochs ---
    # Create a time array with three epochs: past, present, and future
    epochs_years = np.array([epoch - dt, epoch, epoch + dt])
    # Convert the epochs from calendar years to Modified Julian Date (MJD)
    epochs_mjd = [octo.years2mjd(float(y)) for y in epochs_years]

    # --- Simulate astrometric positions at each epoch ---
    # Predict future and past RA/Dec based on position, proper motion, and acceleration
    future_ra = fake_pos(star.ra, star.pm_ra, star.acc_ra, dt)
    future_dec = fake_pos(star.dec, star.pm_dec, star.acc_dec, dt)
    past_ra = fake_pos(star.ra, star.pm_ra, star.acc_ra, -dt)
    past_dec = fake_pos(star.dec, star.pm_dec, star.acc_dec, -dt)

    # --- Propagate errors for simulated data points ---
    # Compute uncertainty in RA/Dec at past and future epochs using PM and acceleration errors
    future_ra_err = propagate_error(ra_err.value, star.sigma_pm_ra, star.sigma_acc_ra, dt)
    future_dec_err = propagate_error(dec_err.value, star.sigma_pm_dec, star.sigma_acc_dec, dt)
    past_ra_err = propagate_error(ra_err.value, star.sigma_pm_ra, star.sigma_acc_ra, -dt)
    past_dec_err = propagate_error(dec_err.value, star.sigma_pm_dec, star.sigma_acc_dec, -dt)

    # --- Relative positions ---
    # Combine past, present, and future positions into arrays
    ra_vals = np.array([past_ra, star.ra, future_ra])
    dec_vals = np.array([past_dec, star.dec, future_dec])
    
    # Subtract cluster center-of-mass (CM) position to get positions relative to CM
    ra_rel = ra_vals - ra_cm_mas
    dec_rel = dec_vals - dec_cm_mas
    # Set measurement errors (central epoch is assumed to be exact here with error ~1e-6)
    ra_errs = [past_ra_err, ra_err.value, future_ra_err]
    dec_errs = [past_dec_err, dec_err.value, future_dec_err]

    return epochs_mjd, ra_rel, dec_rel, ra_errs, dec_errs


# ========================================================
#  Total Angular Accelerations and Uncertainty 
# ========================================================

def total_accelerations(star):
    """
    Compute total plane-of-sky angular and physical acceleration
    (and their uncertainties) for a given star.

    Parameters
    ----------
    star : StarData
        A star object with acc_ra, acc_dec, sigma_acc_ra, and sigma_acc_dec attributes.

    Returns
    -------
    a_total_masyr2 : float
        Total angular acceleration in mas/yr².
    a_total_masyr2_err : float
        Uncertainty in total angular acceleration.
    a_total_kms2 : float
        Total physical acceleration in km/s².
    a_total_kms2_err : float
        Uncertainty in physical acceleration.
    """
    a_ra = star.acc_ra
    a_dec = star.acc_dec
    a_ra_err = star.sigma_acc_ra
    a_dec_err = star.sigma_acc_dec

    # Angular acceleration magnitude
    a_total_masyr2 = np.sqrt(a_ra**2 + a_dec**2)

    # Uncertainty propagation
    a_total_masyr2_err = np.sqrt(
        (a_ra * a_ra_err / a_total_masyr2)**2 +
        (a_dec * a_dec_err / a_total_masyr2)**2
    )

    # Convert to physical acceleration [km/s²]
    a_total_kms2 = utils.masyr2_to_kms2(a_total_masyr2, distance_km=distance_km)
    a_total_kms2_err = utils.masyr2_to_kms2(a_total_masyr2_err, distance_km=distance_km)

    return a_total_masyr2, a_total_masyr2_err, a_total_kms2, a_total_kms2_err


