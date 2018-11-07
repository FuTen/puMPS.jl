
module puMPS

using LinearAlgebra
using TensorOperations
using LinearMaps
using Arpack
using Optim

include("MPS.jl")
using .MPS

export puMPState, rand_puMPState, mps_tensor, num_sites, bond_dim, phys_dim, set_mps_tensor!,
       apply_blockTM_l, blockTM_dense,
       expect_nn, expect,
       canonicalize_left!,
       minimize_energy_local!, vumps_opt!

export MPO_PBC_uniform, MPO_open_uniform, MPO_PBC_uniform_split, MPO_PBC_split

include("states.jl")

export puMPSTvec, tvec_tensor, momentum, mps_tensors, excitations!, excitations, 
       tangent_space_metric_and_MPO, tangent_space_MPO,
       Hn_in_basis, overlap, fidelity

include("tangentspace.jl")

export ising_local_MPO, ising_PBC_MPO, ising_PBC_MPO_split, ising_OBC_MPO, ising_Hn_MPO_split,
       ANNNI_local_MPO, ANNNI_OBC_MPO, ANNNI_PBC_MPO_split, ANNNI_Hn_MPO_split,
       OBF_local_MPO, OBF_OBC_MPO, OBF_PBC_MPO_split, OBF_Hn_MPO_split,
       potts3_local_MPO, potts3_OBC_MPO, potts3_PBC_MPO_split, potts3_Hn_MPO_split

include("models.jl")

end
