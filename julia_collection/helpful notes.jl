"""
julia> using Pkg

julia> Pkg.activate("/home/kenzhayd/projects/kenzhayd/octoJ_env")
  Activating project at `~/projects/kenzhayd/octoJ_env`

julia> using Pigeons
julia> using Octofitter
┌ Info: Welcome to Octofitter v7.0.0 🐙
│ Check for new releases: https://github.com/sefffal/Octofitter.jl/releases/
└ Read the documentation: https://sefffal.github.io/Octofitter.jl/v7.0.0
┌ Info: Note: Julia was started with only one thread. Some models may run faster if you supply multiple threads.
│ To enable multithreading, run:
│
│         julia --threads=auto
│
└ or set the environment variable `JULIA_NUM_THREADS=auto` and restart julia.

julia> pt = Pigeons.PT("/home/kenzhayd/projects/kenzhayd/results/latest"; round=7)
PT("/project/6039459/kenzhayd/results/all/2025-09-05-16-49-42-jjLNb9jv/round=7/checkpoint/results/all/2025-09-05-17-30-48-bozgMx9C")

julia> model = pt.inputs.target
LogDensityModel for System Omega_Cen of dimension 47 and 30 epochs with fields .ℓπcallback and .∇ℓπcallback


julia> results = Octofitter.Chains(model, pt)
Chains MCMC chain (128×79×1 Array{Float64, 3}):

julia> function write_checkpoint(pt)
           if !pt.inputs.checkpoint
               return
           end
           checkpoint_folder = mkpath("$(pt.exec_folder)/round=$(pt.shared.iterators.round)/checkpoint")

           # beginning of serialization session
           flush_immutables!()

           # each process saves its replicas
           for replica in locals(pt.replicas)
               serialize("$checkpoint_folder/replica=$(replica.replica_index).jls", replica)
           end

           only_one_process(pt) do
               serialize("$checkpoint_folder/shared.jls", pt.shared)
               serialize("$checkpoint_folder/reduced_recorders.jls", pt.reduced_recorders)
               # only need to save Inputs & immutables at first round
               if pt.shared.iterators.round == 1
                   serialize("$(pt.exec_folder)/inputs.jls", pt.inputs)
                   # this needs to be last!
                   if !isfile("$(pt.exec_folder)/immutables.jls") # if running via submission, this is written for us
                       serialize_immutables("$(pt.exec_folder)/immutables.jls")
                   end
               end
           end

           # end of serialization session
           flush_immutables!()

           # signal that we are done
           for replica in locals(pt.replicas)
               signal_folder = mkpath("$checkpoint_folder/.signal")
               touch("$signal_folder/finished_replica=$(replica.replica_index)")
           end
       end
write_checkpoint (generic function with 1 method)

julia> Pigeons.write_checkpoint(pt)



[kenzhayd@login1 kenzhayd]$ find /home/kenzhayd/projects/kenzhayd -type d -name "checkpoint"
julia> pt = Pigeons.PT("/home/kenzhayd/projects/kenzhayd/results/all/2025-09-05-16-49-42-jjLNb9jv")

pt = PT("results/latest")
pt = pigeons(pt)





Getting an Octoplot 

[kenzhayd@login1 kenzhayd]$ ls results/all
[kenzhayd@login1 kenzhayd]$ ls results/all/2025-09-06-17-45-26-Hgo4ykNq
 immutables.jls  'round=1'   'round=11'  'round=2'  'round=4'  'round=6'  'round=8'
 inputs.jls      'round=10'  'round=12'  'round=3'  'round=5'  'round=7'  'round=9'

julia> using Pkg

julia> Pkg.activate("/home/kenzhayd/projects/kenzhayd/octoJ_env")
  Activating project at `~/projects/kenzhayd/octoJ_env`

julia> using Octofitter, Pigeons
┌ Info: Welcome to Octofitter v7.0.0 🐙
│ Check for new releases: https://github.com/sefffal/Octofitter.jl/releases/
└ Read the documentation: https://sefffal.github.io/Octofitter.jl/v7.0.0
┌ Info: Note: Julia was started with only one thread. Some models may run faster if you supply multiple threads.
│ To enable multithreading, run:
│
│         julia --threads=auto
│
└ or set the environment variable `JULIA_NUM_THREADS=auto` and restart julia.

julia> pt = Pigeons.PT("results/all/2025-09-06-17-45-26-Hgo4ykNq")
PT("/project/6039459/kenzhayd/results/all/2025-09-07-11-03-34-Sp1NXbMs")

julia> model = pt.inputs.target
LogDensityModel for System Omega_Cen of dimension 47 and 30 epochs with fields .ℓπcallback and .∇ℓπcallback
julia> results = Chains(model, pt)

julia> using CairoMakie, PairPlots

julia> Octofitter.octocorner(model, results)





mulitprocessing in python
https://superfastpython.com/multiprocessing-pool-python/

"""