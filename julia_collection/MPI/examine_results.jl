
"""
Get the Info from Pigeons MPI run
"""

# # If still in current session, just pass the `pt` object:
# results = Chains(model, pt)

# Else, if the sampling has been running in the background, run:
pt = PT(mpi_run)
model = pt.inputs.target
results = Chains(model, pt)


octocorner(model, results, small=true)
# run pigeons
results = Chains(model, pt)

# save chain
Octofitter.savechain("/home/kenzhayd/projects/kenzhayd/mpi_chain_$(n_rounds)", results)

# === 7. Corner Plot ===
corner_plot = octocorner(model, results; small=true)
corner_filename = "/home/kenzhayd/projects/kenzhayd/corner_plot_$(n_rounds).png"
save(corner_filename, corner_plot)

# === 8. Orbit Plot ===
orbit_plot = octoplot(model, results; show_physical_orbit=true, colorbar=true)
orbit_filename = "/home/kenzhayd/projects/kenzhayd/orbit_plot_v1_$(n_rounds).png"
save(orbit_filename, orbit_plot)

# === 9. Orbit Plot again ===

ts = Octofitter.range(54600, 55700, length=200) 
orbit_plot_2 = octoplot(model, results; show_physical_orbit=true, colorbar=false, figscale=1.5, ts=ts)

# Access and modify specific axes
ax_orbit = orbit_plot_2.content[1]  # First axis (usually the orbit plot)
xlims!(ax_orbit, -200, 200)  # Set x-axis limits in mas
ylims!(ax_orbit, -100, 100)  # Set y-axis limits in mas

# Add a title
ax_orbit.title = "Orbits of Fast-Moving Stars in 𝜔 Cen"

# Build filename 
#timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
orbit_filename_2 = "/home/kenzhayd/projects/kenzhayd/orbit_plot_v2_$(n_rounds).png"

save(orbit_filename_2, orbit_plot_2)