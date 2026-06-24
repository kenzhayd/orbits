"""
Orbital Models of High Velocity Stars in Omega Centauri
Using Octofitter
"""

# Environment variables
ENV["JULIA_NUM_THREADS"] = "auto"
ENV["OCTOFITTERPY_AUTOLOAD_EXTENSIONS"] = "yes"

using Octofitter
using Octofitter: @variables, System
using CairoMakie
using PairPlots
using Distributions
using Unitful
using UnitfulAstro
using LinearAlgebra
using Statistics
using Dates
using Pigeons

# Add the directory to LOAD_PATH 
push!(LOAD_PATH, raw"/home/kenzhayd/projects/kenzhayd")
using octo_utils_julia  # local module

# === 1. Select stars and time config ===
star_names = ["A", "B", "C", "D", "E", "F", "G"]
epoch = 2010.0
dt = 1.0

# Dictionaries to store simulation results and likelihood objects
epochs_mjd = Dict{String, Any}()
ra_rel = Dict{String, Any}()
dec_rel = Dict{String, Any}()
ra_errs = Dict{String, Any}()
dec_errs = Dict{String, Any}()
astrom_likelihoods = Dict{String, Any}()

# === 2. Simulate astrometry and create likelihood objects ===
for name in star_names
    star = octo_utils_julia.stars[name]

    emjd, ra_r, dec_r, ra_e, dec_e = octo_utils_julia.simulate_astrometry(star, epoch, dt)

    epochs_mjd[name] = emjd
    ra_rel[name] = ra_r
    dec_rel[name] = dec_r
    ra_errs[name] = ra_e
    dec_errs[name] = dec_e

    obs = ntuple(i -> (
        epoch = emjd[i],
        ra = ra_r[i],
        dec = dec_r[i],
        σ_ra = ra_e[i],
        σ_dec = dec_e[i],
        cor = 0.0
    ), length(emjd))

    astrom_likelihoods[name] = PlanetRelAstromLikelihood(obs; name = name)
end

# === 3. Define companions ===
planet_1 = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["A"])],
    variables = @variables begin
        M = system.M
        P ~ Uniform(1, 200000)         # Period in yrs
        a = cbrt(M * P^2)     # Semi-Major axis in AU
        e ~ Uniform(0.0, 0.99)         # Eccentricity
        i ~ Sine()                     # Inclination [rad]
        ω ~ UniformCircular()          # Argument of periastron [rad]
        Ω ~ UniformCircular()          # Longitude of ascending node [rad]
        θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
        tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
    end
)
planet_3 = Planet(
    name = "C",
    basis = Visual{KepOrbit},
    likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["C"])],
    variables =@variables begin
        M = system.M
        P ~ Uniform(1, 200000)         # Period in yrs
        a = cbrt(M * P^2)     # Semi-Major axis in AU
        e ~ Uniform(0.0, 0.99)         # Eccentricity
        i ~ Sine()                     # Inclination [rad]
        ω ~ UniformCircular()          # Argument of periastron [rad]
        Ω ~ UniformCircular()          # Longitude of ascending node [rad]
        θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
        tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M) 
    end
)

planet_4 = Planet(
    name = "D",
    basis = Visual{KepOrbit},
    likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["D"])],
    variables =@variables begin
        M = system.M
        P ~ Uniform(1, 200000)         # Period in yrs
        a = cbrt(M * P^2)     # Semi-Major axis in AU
        e ~ Uniform(0.0, 0.99)         # Eccentricity
        i ~ Sine()                     # Inclination [rad]
        ω ~ UniformCircular()          # Argument of periastron [rad]
        Ω ~ UniformCircular()          # Longitude of ascending node [rad]
        θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
        tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
    end
)

planet_5 = Planet(
    name = "E",
    basis = Visual{KepOrbit},
    likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["E"])],
    variables =@variables begin
        M = system.M
        a ~ Uniform(1, 100000)
        e ~ Uniform(0.0, 0.99)
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω,M=M)  
    end
)

planet_6 = Planet(
    name = "F",
    basis = Visual{KepOrbit},
    likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["F"])],
    variables =@variables begin
        M = system.M
        a ~ Uniform(1, 100000)
        e ~ Uniform(0.0, 0.99)
        i ~ Sine()
        ω ~ UniformCircular()
        Ω ~ UniformCircular()
        θ ~ UniformCircular()
        tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)  
    end
)

# Note stars B and G were not used in the Hablerle et al. 2024 paper

# planet_2 = Planet(
#     name = "B",
#     basis = Visual{KepOrbit},
#     likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["B"])],
#     variables = @variables begin
#           M = system.M    # Host mass [solar masses]
#           P ~ Uniform(1, 200000)         # Period in yrs
#           a = cbrt(M * P^2)     # Semi-Major axis in AU
#           e ~ Uniform(0.0, 0.99)         # Eccentricity
#           i ~ Sine()                     # Inclination [rad]
#           ω ~ UniformCircular()          # Argument of periastron [rad]
#           Ω ~ UniformCircular()          # Longitude of ascending node [rad]
#           θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
#           tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)
# )

#
# planet_7 = Planet(
#     name = "G",
#     basis = Visual{KepOrbit},
#     likelihoods = [ObsPriorAstromONeil2019(astrom_likelihoods["G"])],
#     variables = @variables begin
#           M = system.M
#           P ~ Uniform(1, 200000)         # Period in yrs
#           a = cbrt(M * P^2)     # Semi-Major axis in AU
#           e ~ Uniform(0.0, 0.99)         # Eccentricity
#           i ~ Sine()                     # Inclination [rad]
#           ω ~ UniformCircular()          # Argument of periastron [rad]
#           Ω ~ UniformCircular()          # Longitude of ascending node [rad]
#           θ ~ UniformCircular()          # Mean anomaly at reference epoch [rad]
#           tp = θ_at_epoch_to_tperi(θ, 55197.0; a=a, e=e, i=i, ω=ω, Ω=Ω, M=M)  
#     end
# )

# === 4. Define the full system ===
sys = System(
    name = "Omega_Cen",
    likelihoods = [],
    companions = [planet_1,planet_3, planet_4, planet_5, planet_6],
    variables = @variables begin
        plx ~ truncated(Normal(0.19, 0.004), lower=0)  # Parallax [mas]
        M ~ Uniform(100, 200000)    # Host mass [solar masses]
    end
)

# === 5. Model ===
model = Octofitter.LogDensityModel(sys)

# === 6. Fit with Pigeons ===
chain, pt = octofit_pigeons(model; n_rounds=16, n_chains=60, n_chains_variational=60)
println(chain)

# Save Chain 
chain_name = "16_rounds_chain"
Octofitter.savechain("/home/kenzhayd/projects/kenzhayd/$(chain_name)", chain)

# === 7. Corner Plot ===
corner_plot = octocorner(model, chain; small=true)
corner_plot_name = "octo_corner_16"
corner_filename = "/home/kenzhayd/projects/kenzhayd/$(corner_plot_name)_Julia.png"
save(corner_filename, corner_plot)

# === 8. Orbit Plot ===
orbit_plot = octoplot(model, chain; show_physical_orbit=true, colorbar=true)
orbit_plot_name = "orbit_v1_16rounds"
orbit_filename = "/home/kenzhayd/projects/kenzhayd/$(orbit_plot_name)_Julia.png"
save(orbit_filename, orbit_plot)

# === 9. Orbit Plot again ===

ts = Octofitter.range(54600, 55700, length=200) 
orbit_plot_2 = octoplot(model, chain; show_physical_orbit=true, colorbar=false, figscale=1.5, ts=ts)

# Access and modify specific axes
ax_orbit = orbit_plot_2.content[1]  # First axis (usually the orbit plot)
xlims!(ax_orbit, -200, 200)  # Set x-axis limits in mas
ylims!(ax_orbit, -100, 100)  # Set y-axis limits in mas

# Add a title
ax_orbit.title = "Orbits of Fast-Moving Stars in 𝜔 Cen"

# Build filename 
orbit_plot_name_2 = "orbit_v2_16_rounds" 
#timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
orbit_filename_2 = "/home/kenzhayd/projects/kenzhayd/$(orbit_plot_name_2)_Julia.png"

save(orbit_filename_2, orbit_plot_2)

