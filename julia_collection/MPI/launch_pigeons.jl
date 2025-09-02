# to run julia launch_pigeons.jl distributed_octo_orbit.jl 70
using Pkg
Pkg.activate("/home/kenzhayd/projects/kenzhayd/octoJ_env")

# Make sure Pigeons and HDF5 are installed
Pkg.add(["Pigeons", "HDF5", "Preferences"])
using Preferences, HDF5, Pigeons

set_preferences!(
    HDF5,
    "libhdf5" => ENV["EBROOTHDF5"]*"/lib/libhdf5_hl.so",
    "libhdf5_hl" => ENV["EBROOTHDF5"]*"/lib/libhdf5_hl.so",
    force = true
)

modelfname = ARGS[1]
n_proc = parse(Int, ARGS[2])

n_chains = 24

Pigeons.setup_mpi(
    submission_system = :slurm,
    environment_modules = ["StdEnv/2023", "intel", "openmpi", "julia/1.10", "hdf5"],
    library_name = ENV["EBROOTOPENMPI"]*"/lib/libmpi",
    add_to_submission = [
        "#SBATCH --output=octo_run_%j.out",
        "#SBATCH --error=octo_run_%j.err",
        "#SBATCH --time=48:00:00",
        "#SBATCH --account=def-vhenault",
        "#SBATCH --nodes=1",           # force single node
        "#SBATCH --ntasks=$n_chains", # one MPI process per chain
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem-per-cpu=8G"
    ]
)
println("Setup MPIProcesses")

include("distributed_octo_orbit.jl")

traces = []
round_trip = []


pt = pigeons(
    target = Pigeons.LazyTarget(model),
    record = [traces; round_trip; record_default()],
    on = Pigeons.MPIProcesses(
        n_mpi_processes = n_chains,   # one MPI process per chain
        n_threads = 1,
        dependencies = [abspath("distributed_octo_orbit.jl")]
    )
)

