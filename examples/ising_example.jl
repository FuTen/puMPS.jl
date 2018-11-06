using puMPS

D = 8
N = 32

M = rand_puMPState(ComplexF64, 2, D, N)

minimize_energy_local!(M, ising_local_MPO(ComplexF64), 1000, step=0.1)

println("Computing excitations!")

ks_tocompute = [-2,-1,0,1,2]
num_states = [5,4,7,4,5]

ens, ks, exs = excitations!(M, ising_PBC_MPO_split(ComplexF64), ks_tocompute, num_states)

H1 = Hn_in_basis(M, ising_Hn_MPO_split(ComplexF64, 1, N), exs, ks_tocompute)
H2 = Hn_in_basis(M, ising_Hn_MPO_split(ComplexF64, 2, N), exs, ks_tocompute)

ind1 = argmin(real.(ens))
indT = argmax(abs.(H2[:,ind1]))

en1 = ens[ind1]
enT = ens[indT]

fac = 2.0 / (enT-en1)

c = 2*abs2(H2[indT,ind1] * fac)
@show real(c)

using PyPlot

plot(ks, real(ens), "o")
show()