
include("../src/MPS.jl")

include("../src/puMPS.jl")

using MPS, puMPS

D = 8
N = 32

M = rand_puMPState(Complex128, 2, D, N)

minimize_energy_local!(M, ising_local_MPO(Complex128), 100, step=0.1)

println("Computing excitations!")

ens, ks, exs = excitations!(M, ising_PBC_MPO_split(Complex128), [-2,-1,0,1,2], [5,4,7,4,5])

using PyPlot

plot(ks, real(ens), "o")
show()