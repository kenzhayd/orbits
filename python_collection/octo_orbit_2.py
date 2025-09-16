"""
Orbital Models of High Velocity Stars in Omega Centauri
Using Octofitterpy
"""

#Environment Variables
import os
os.environ["JULIA_NUM_THREADS"] = "auto"
os.environ["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"


#Imports 
import numpy as np
import astropy.units as u
from astropy.time import Time
import astropy.constants as const
import pandas as pd
from astropy.table import Table
from datetime import datetime
from scipy import stats


import octo_utils_2 as utils

import octofitterpy as octo

# === 1. fast-moving stars and epoch configuration ===
star_names = ["A", "B","C","D","E","F","G"]
epoch = 2010.0
dt = 1

# Dictionaries to for astrometric data and likelihood objects
epochs_mjd = {}
ra_rel = {}
dec_rel = {}
ra_errs = {}
dec_errs = {}

astrom_likelihoods = {}

# === 2. Create likelihood objects ===
for name in star_names:
    star = utils.stars[name]
    emjd, ra_r, dec_r, ra_e, dec_e = utils.simulate_astrometry(star=star, epoch=epoch, dt=dt)
    
    epochs_mjd[name] = emjd
    ra_rel[name] = ra_r
    dec_rel[name] = dec_r
    ra_errs[name] = ra_e
    dec_errs[name] = dec_e

    astrom_likelihoods[name] = octo.PlanetRelAstromLikelihood(
        epoch=epochs_mjd[name],             # Observation epochs (Modified Julian Dates)
        ra=ra_rel[name].tolist(),           # Relative Right Ascension values [mas]
        dec=dec_rel[name].tolist(),         # Relative Declination values [mas]
        σ_ra=ra_errs[name],                 # Measurement uncertainties in RA [mas]
        σ_dec=dec_errs[name],               # Measurement uncertainties in Dec [mas]
        cor=[0.0] * len(epochs_mjd[name])   # RA/Dec error correlation (zero for all epochs)
    )

# Access via astrom_likelihoods["A"]
# Check Errors: print(dec_errs["A"])

"""
 Companions with "observable based priors"
""" 
# "observable based priors" of K. O'Neil 2019 "Improving Orbit Estimates for Incomplete Orbits with a New Approach to Priors: with Applications from Black Holes to Planets".
# Supply Uniform priors on all Campbell orbital parameters and a Uniform prior on Period (not semi-major axis). 
# This period prior has a significant impact in the fit and recommendations for its range were not published in the original paper.
# Paper: https://arxiv.org/pdf/1809.05490

#Note: Stars B and G were not used in the Hablerle et al. 2024 paper
planet_1 = octo.Planet(
    name = "A",
    basis = "Visual{KepOrbit}",
    likelihoods = [octo.ObsPriorAstromONeil2019(astrom_likelihoods["A"]), astrom_likelihoods["A"]],
    priors = 
    """
          P ~ Uniform(1, 2000000)         # Period in yrs
          a = cbrt(system.M * A.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, A, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """
)
planet_2 = octo.Planet(
    name = "B",                      # Name of the companion (used in output labels)
    basis = "Visual{KepOrbit}",              # Keplerian orbital basis for visual binary
    likelihoods=[octo.ObsPriorAstromONeil2019(astrom_likelihoods["B"]), astrom_likelihoods["B"]], 
    priors =
    """
          P ~ Uniform(1, 2000000)        # Period in yrs
          a = cbrt(system.M * B.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, B, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """
)
planet_3 = octo.Planet(
    name = "C",                              # Name of the companion (used in output labels)
    basis = "Visual{KepOrbit}",              # Keplerian orbital basis for visual binary
    likelihoods=[octo.ObsPriorAstromONeil2019(astrom_likelihoods["C"]), astrom_likelihoods["C"]], 
    priors =
    """
          P ~ Uniform(1, 2000000)         # Period in yrs
          a = cbrt(system.M * C.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, C, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """
)
planet_4 = octo.Planet(
    name = "D",                              # Name of the companion (used in output labels)
    basis = "Visual{KepOrbit}",              # Keplerian orbital basis for visual binary
    likelihoods=[octo.ObsPriorAstromONeil2019(astrom_likelihoods["D"]), astrom_likelihoods["D"]], 
    priors =
    """
          P ~ Uniform(1, 2000000)         # Period in yrs
          a = cbrt(system.M * D.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, D, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """
)
planet_5 = octo.Planet(
    name = "E",                              # Name of the companion (used in output labels)
    basis = "Visual{KepOrbit}",              # Keplerian orbital basis for visual binary
    likelihoods=[octo.ObsPriorAstromONeil2019(astrom_likelihoods["E"]), astrom_likelihoods["E"]], 
    priors =
    """
          P ~ Uniform(1, 2000000)         # Period in yrs
          a = cbrt(system.M * E.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, E, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """
)
planet_6 = octo.Planet(
    name = "F",                              # Name of the companion (used in output labels)
    basis = "Visual{KepOrbit}",              # Keplerian orbital basis for visual binary
    likelihoods=[octo.ObsPriorAstromONeil2019(astrom_likelihoods["F"]), astrom_likelihoods["F"]],
    priors =
    """
          P ~ Uniform(1, 2000000)         # Period in yrs
          a = cbrt(system.M * F.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, F, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """ 
)
planet_7 = octo.Planet(
    name = "G",                              # Name of the companion (used in output labels)
    basis = "Visual{KepOrbit}",              # Keplerian orbital basis for visual binary
    likelihoods=[octo.ObsPriorAstromONeil2019(astrom_likelihoods["G"]), astrom_likelihoods["G"]], 
    priors =
    """
          P ~ Uniform(1, 2000000)         # Period in yrs
          a = cbrt(system.M * G.P^2)     # Semi-Major axis in AU
          e ~ Uniform(0.0, 0.99)         # Eccentricity
          i ~ Sine()                     # Inclination [rad]
          ω ~ UniformCircular()          # Argument of periastron [rad]
          Ω ~ UniformCircular()          # Longitude of ascending node [rad]
          θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
          tp = θ_at_epoch_to_tperi(system, G, 55197.0)  # Reference MJD for θ (adjust to match your data)
      """
)


"""
Orbit Fitting and Plotting with Octofitter
"""
# Pigeons Algorithm: https://arxiv.org/abs/2308.09769


# === 3. Define the full system ===
# The system includes the host star’s mass and parallax, and any orbiting companions.
# Note: Stars B and G were not used in the Hablerle et al. 2024 paper
# Stars used by Hablerle et al. 2024: planet_1,planet_3,planet_4,planet_5,planet_6

sys = octo.System(
    name = "Omega_Cen",             # System name (used for output files/plot titles)
    priors = 
    """
        M   ~ Uniform(100, 200000)    # Host mass [solar masses]
        plx ~ truncated(Normal(0.19, 0.004), lower=0)     # Parallax [mas]
    """,
    likelihoods = [],                 # No system-level likelihoods in this case
    companions = [planet_1]             # List of orbiting bodies
) 



# === 4. Log-probability model ===
model = octo.LogDensityModel(sys)

# === 5. Fit the model ===
chain, pt = octo.octofit_pigeons(model, n_rounds= 12, n_chains= 30, n_chains_variational= 30)
print(chain)

# Generate the corner plot
# Cluster location: /home/kenzhayd/projects/def-vhenault/kenzhayd/
companions = "A_ObsPrior_12rounds_30chains"
corner_plot_name = rf"C:\Users\macke\OneDrive - Saint Marys University\Summer Research 2025\octo_orbit\octo_plots\corner_{companions}.png"
corner_plot = octo.octocorner(model, chain, small=True, fname=corner_plot_name)

# Generate the orbit plot
# Cluster location: /home/kenzhayd/projects/def-vhenault/kenzhayd/

# Time scale
ts = octo.Octofitter.range(54600, 55700, length=200) 

orbit_plot_name = rf"C:\Users\macke\OneDrive - Saint Marys University\Summer Research 2025\octo_orbit\octo_plots\orbit_{companions}.png"
orbit_plot = octo.octoplot(model,chain, show_physical_orbit = True, colorbar = True, ts = ts, fname=orbit_plot_name)

orbit_plot_2_name = rf"C:\Users\macke\OneDrive - Saint Marys University\Summer Research 2025\octo_orbit\octo_plots\orbit_v2_{companions}.png"
orbit_plot_2 = octo.octoplot(model,chain, show_physical_orbit = True, colorbar = False, ts = ts, fname=orbit_plot_2_name)


# Total system mass (usually primary + companion) in solar masses
print("Total Mass [Solar Masses]:", np.percentile(chain["M"], [16, 50, 84]))
print("Lower Mass Limit [Solar Masses]:", np.percentile(chain["M"], [1]))