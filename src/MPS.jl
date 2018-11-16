
module MPS

using LinearAlgebra
using TensorOperations
using LinearMaps
using KrylovKit

export MPSTensor, bond_dim, bond_dim_L, bond_dim_R, phys_dim, mps_tensor, mps_tensor_shape, 
       rand_MPSTensor_unitary, rand_MPSTensor,
       MPS_TM, TM_dense, eyeTM, TM_dense_op_nn, 
       applyTM_op_nn_l, applyTM_l, applyTM_r,
       workvec_applyTM_l, workvec_applyTM_l!, res_applyTM_l, res_applyTM_l!, applyTM_l!, 
       workvec_applyTM_r, workvec_applyTM_r!, res_applyTM_r, res_applyTM_r!, applyTM_r!, 
       tm_eigs_sparse, tm_eigs_dense, tm_eigs, tm_dominant_eigs,
        gauge_transform, canonicalize_left,
       MPOTensor, IdentityMPOTensor, mpo_tensor_shape, rand_MPOTensor,
       mul_MPO, MPO_open, MPS_MPO_TM, TM_dense_MPO, applyTM_MPO_l, applyTM_MPO_r,
       worklen_applyTM_MPO_l, workvec_applyTM_MPO_l, workvec_applyTM_MPO_l!, 
       res_applyTM_MPO_l, res_applyTM_MPO_l!, applyTM_MPO_l!,
       workvec_applyTM_MPO_r, workvec_applyTM_MPO_r!, applyTM_MPO_r!, res_applyTM_MPO_r, res_applyTM_MPO_r!,
       TM_convert

"""
We will use the convention that the middle index is the physical one,
while the first and third are the left and right virtual indices.
In pictures, for an `MPSTensor` `A`:

 1--A--3
    |
    2

"""
MPSTensor{T} = Array{T,3}

bond_dim_L(A::MPSTensor) = size(A,1)
bond_dim_R(A::MPSTensor) = size(A,3)
bond_dim(A::MPSTensor) = bond_dim_L(A)
phys_dim(A::MPSTensor) = size(A,2)
mps_tensor_shape(d::Integer, D::Integer) = (D,d,D)
mps_tensor_shape(d::Integer, D1::Integer, D2::Integer) = (D1,d,D2)

"""
MPO tensor:

     2
     |
 1---o---3
     |
     4
"""
MPOTensor{T} = Array{T,4}

struct IdentityMPOTensor
end

mpo_tensor_shape(d::Integer, M::Integer) = (M,d,M,d)
mpo_tensor_shape(d::Integer, M1::Integer, M2::Integer) = (M1,d,M2,d)

function randunitary(::Type{T}, N::Integer)::Matrix{T} where {T}
    if T <: Complex
        rT = real(T)
        A = complex.(randn(rT, N,N), randn(rT, N,N)) / √2
    else
        A = randn(T, N,N)
    end
    Q, R = qr(A)
    r = diag(R)
    L = diagm(0 => r ./ abs.(r))
    Q*L
end

#Note: An MPS generated this way already has r proportional to I and largest tm eigenvalue = 1
function rand_MPSTensor_unitary(::Type{T}, d::Integer, D1::Integer, D2::Integer)::MPSTensor{T} where {T}
    U = randunitary(T, max(D1,d*D2))
    reshape(U[1:D1,1:d*D2], mps_tensor_shape(d,D1,D2))
end
rand_MPSTensor_unitary(T, d::Integer, D::Integer) = rand_MPSTensor_unitary(T, d, D, D)

function rand_MPSTensor(::Type{T}, d::Integer, D1::Integer, D2::Integer)::MPSTensor{T}  where {T}
    shp = mps_tensor_shape(d,D1,D2)
    if T <: Complex
        rT = real(T)
        A = complex.(randn(rT, shp), randn(rT, shp)) / √2
    else
        A = randn(T, shp)
    end
    A
end
rand_MPSTensor(T, d::Integer, D::Integer) = rand_MPSTensor(T, d, D, D)

function rand_MPOTensor(::Type{T}, d::Integer, M1::Integer, M2::Integer)::MPOTensor{T}  where {T}
    shp = mpo_tensor_shape(d,M1,M2)
    if T <: Complex
        rT = real(T)
        O = complex.(randn(rT, shp), randn(rT, shp)) / √2
    else
        O = randn(T, shp)
    end
    O
end
rand_MPOTensor(T, d::Integer, M::Integer) = rand_MPOTensor(T,d,M,M)

"""
Dense transfer matrix with MPO:

 1----A-----4
      |
 2----o-----5
      |
 3--conj(B)-6
"""
MPS_MPO_TM{T} = Array{T,6}

"""
Dense transfer matrix without MPO:

 1----A-----3
      |
 2--conj(B)-4

"""
MPS_TM{T} = Array{T,4}

#Note: We will generally use MPS_MPO_TM for function arguments, providing functions to convert where possible.

LinearAlgebra.tr(TM::MPS_MPO_TM{T}) where {T} = scalar(@tensor res[] := TM[t,m,b,t,m,b])

#Turn the MPS TM into an MPS TM with (size 1) MPO indices
TM_convert(TM::MPS_TM{T}) where {T} = reshape(TM, (size(TM,1),1,size(TM,2), size(TM,3),1,size(TM,4)))::MPS_MPO_TM{T}

function TM_convert(TM::MPS_MPO_TM{T})::MPS_TM{T} where {T}
    size(TM,2) == size(TM,5) == 1 || error("MPO bond dimensions not equal to 1!")
    dropdims(TM, dims=(2,5))
end

function TM_dense(A::MPSTensor, B::MPSTensor)::MPS_MPO_TM
    GC.enable(false)
    @tensor TM[lA,lB,rA,rB] := A[lA, s, rA] * conj(B[lB, s, rB]) #NOTE: This will generally allocate to do permutations!
    GC.enable(true)
    GC.gc(false)
    TM_convert(TM)
end

eyeTM(T::Type, D::Integer)::MPS_MPO_TM = (e = Matrix{T}(I,D,D); reshape(kron(e,e), (D,1,D, D,1,D)))

function TM_dense_op_nn(A1::MPSTensor, A2::MPSTensor, B1::MPSTensor, B2::MPSTensor, op::Array{T,4})::MPS_MPO_TM where {T}
    @tensor TM[lA1,lB1,rA2,rB2] := ((A1[lA1, p1k, iA] * A2[iA, p2k, rA2]) * op[p1b, p2b, p1k, p2k]) * (conj(B1[lB1, p1b, iB]) * conj(B2[iB, p2b, rB2])) #NOTE: Allocates intermediate arrays
    TM_convert(TM)
end

function applyTM_op_nn_l(A1::MPSTensor, A2::MPSTensor, B1::MPSTensor, B2::MPSTensor, op::Array{T,4}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    @tensor TM21[k1,b1,k4,b4] := ((((TM2[k1,b1,k2,b2] * A1[k2,s1,k3]) * op[t1,t2,s1,s2]) * conj(B1[b2,t1,b3])) * A2[k3,s2,k4]) * conj(B2[b3,t2,b4])
    TM_convert(TM21)
end

#This is a workaround for StridedArray not including reshaped arrays formed from a FastContiguousSubArray in in Julia <= 0.6.
#Since TensorOperations uses the StridedArray type, it misses these cases. Here, we use pointer manipulation to generate
#views of sections of a vector with an arbitrary "size".
function unsafe_reshaped_subvec(w::Vector{T}, ind1::Integer, sz::NTuple) where {T}
    len = prod(sz)
    @assert length(w) >= ind1+len-1 "Vector is not big enough to hold subarray by $(ind1+len-1 - length(w)) elements!"
    unsafe_wrap(Array, pointer(w, ind1), sz)
end

function res_applyTM_l(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    zeros(T, (size(TM2,1), 1, size(TM2,2), bond_dim_R(A1), 1, bond_dim_R(B1)))
end

function res_applyTM_l!(res::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    sz = (size(TM2,1), 1, size(TM2,2), bond_dim_R(A1), 1, bond_dim_R(B1))
    size(res) == sz ? res : zeros(T, sz)
end

workvec_applyTM_l(A1::MPSTensor{T}, B1::MPSTensor{T}) where {T} = Vector{T}(undef, bond_dim(A1)^2 * bond_dim(B1)^2 * phys_dim(A1) * 2)

function worklen_applyTM_l(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    len1 = size(TM2,1) * size(TM2,2) * size(TM2,3) * size(B1,2) * size(B1,3)
    len2 = prod(size(TM2)[1:2]) * size(B1)[3] * size(A1,3)
    max(2len1, len1 + len2)
end

function workvec_applyTM_l(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    Vector{T}(undef, worklen_applyTM_l(A1, B1, TM2))
end

function workvec_applyTM_l!(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    len = worklen_applyTM_l(A1, B1, TM2)
    work = length(work) < len ? resize!(work, len) : work
end

function applyTM_l!(TM21::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}, work::Vector{T}) where {T}
    TM2 = TM_convert(TM2)
    TM21c = TM_convert(TM21)

    wsz1 = (size(TM2)[1:3]..., size(B1)[2:3]...)
    wlen1 = prod(wsz1)
    TM2B1 = unsafe_reshaped_subvec(work, 1, wsz1)
    @tensor TM2B1[lA,lB, mA,s,rB] = TM2[lA,lB, mA,mB] * conj(B1[mB,s,rB])
    
    wsz2 = (wsz1[1:2]..., wsz1[5], wsz1[3:4]...)
    wlen2 = prod(wsz2)
    TM2B1p = unsafe_reshaped_subvec(work, wlen1+1, wsz2) #take work vector section after TM2B1 storage
    @tensor TM2B1p[lA,lB, rB,mA,s] = TM2B1[lA,lB, mA,s,rB]
    TM2B1p_copy = unsafe_reshaped_subvec(work, 1, wsz2)
    copyto!(TM2B1p_copy, TM2B1p)
    TM2B1p = TM2B1p_copy
    
    TM2sz = size(TM2)
    wsz3 = (TM2sz[1:2]..., size(B1)[3], size(A1,3))
    TM21p = unsafe_reshaped_subvec(work, wlen2+1, wsz3)
    @tensor TM21p[lA,lB, rB,rA] = TM2B1p[lA,lB, rB,mA,s] * A1[mA,s,rA]
    @tensor TM21c[lA,lB, rA,rB] = TM21p[lA,lB, rB,rA]
    
    TM21
end

function applyTM_l(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    GC.enable(false)
    @tensor TM21[lA,lB, rA,rB] := (TM2[lA,lB, mA,mB] * conj(B1[mB,s,rB])) * A1[mA,s,rA] #NOTE: Intermediates, final permutation.
    GC.enable(true)
    GC.gc(false)
    TM_convert(TM21)
end

function applyTM_l(A::MPSTensor, B::MPSTensor, x::Matrix)
    GC.enable(false)
    @tensor res[a, d] := conj(A[b, s, a]) * (x[b, c] * B[c, s, d]) #NOTE: This requires at least one intermediate array
    GC.enable(true)
    GC.gc(false)
    res
end
applyTM_l(A::MPSTensor, B::MPSTensor, x::Vector) = applyTM_l(A, B, reshape(x, (bond_dim(A), bond_dim(B))))

function applyTM_r(A::MPSTensor, B::MPSTensor, x::Matrix)
    @tensor res[a, d] := (A[a, s, b] * x[b, c]) * conj(B[d, s, c]) #NOTE: This requires at least one intermediate array
end
applyTM_r(A::MPSTensor, B::MPSTensor, x::Vector) = applyTM_r(A, B, reshape(x, (bond_dim(A), bond_dim(B))))

function applyTM_r(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    GC.enable(false)
    @tensor TM12[lA,lB, rA,rB] := conj(B1[lB,s,mB]) * (A1[lA,s,mA] * TM2[mA,mB, rA,rB]) #NOTE: Intermediates, final permutation.
    GC.enable(true)
    GC.gc(false)
    TM_convert(TM12)
end

function applyTM_r!(TM12::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}, work::Vector{T}) where {T}
    TM2 = TM_convert(TM2)
    TM12c = TM_convert(TM12)

    wsz1 = (size(A1)[1:2]..., size(TM2)[2:4]...)
    wlen1 = prod(wsz1)
    A1TM2 = unsafe_reshaped_subvec(work, 1, wsz1)
    @tensor A1TM2[lA,s,mB, rA,rB] = A1[lA,s,mA] * TM2[mA,mB, rA,rB]
    
    wsz2 = (wsz1[2:3]..., wsz1[1], wsz1[4:5]...)
    wlen2 = prod(wsz2)
    A1TM2p = unsafe_reshaped_subvec(work, wlen1+1, wsz2) #take work vector section after TM2B1 storage
    @tensor A1TM2p[s,mB,lA, rA,rB] = A1TM2[lA,s,mB, rA,rB]
    A1TM2p_copy = unsafe_reshaped_subvec(work, 1, wsz2)
    copyto!(A1TM2p_copy, A1TM2p)
    A1TM2p = A1TM2p_copy
    
    TM12sz = size(TM12c)
    wsz3 = (TM12sz[2], TM12sz[1], TM12sz[3:4]...)
    TM12p = unsafe_reshaped_subvec(work, wlen2+1, wsz3)
    @tensor TM12p[lB,lA, rA,rB] = conj(B1[lB,s,mB]) * A1TM2p[s,mB,lA, rA,rB]

    @tensor TM12c[lA,lB, rA,rB] = TM12p[lB,lA, rA,rB]
    
    TM12
end

function res_applyTM_r(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    zeros(T, (bond_dim_L(A1), 1, bond_dim_L(B1), size(TM2,3), 1, size(TM2,4)))
end

function res_applyTM_r!(res::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    sz = (bond_dim_L(A1), 1, bond_dim_L(B1), size(TM2,3), 1, size(TM2,4))
    size(res) == sz ? res : zeros(T, sz)
end

workvec_applyTM_r(A1::MPSTensor{T}, B1::MPSTensor{T}) where {T} = Vector{T}(bond_dim(A1)^2 * bond_dim(B1)^2 * phys_dim(A1) * 2)

function worklen_applyTM_r(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    TM2 = TM_convert(TM2)
    len1 = prod((size(A1)[1:2]..., size(TM2)[2:4]...))
    len2 = prod((size(A1,1) * size(B1,1), size(TM2)[3:4]...))
    max(2len1, len1 + len2)
end

function workvec_applyTM_r(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    Vector{T}(undef, worklen_applyTM_r(A1, B1, TM2))
end

function workvec_applyTM_r!(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    len = worklen_applyTM_r(A1, B1, TM2)
    work = length(work) < len ? resize!(work, len) : work
end

function tm_eigs_sparse(A::MPSTensor{T}, B::MPSTensor{T}, dirn::Symbol, nev::Integer=1; 
                    x0::Matrix{T}=ones(T,(bond_dim(A), bond_dim(B)))) where {T}
    @assert dirn in (:L, :R)
    
    D = bond_dim(A)
    DB = bond_dim(B)
    
    x = zeros(T, (D,DB))
    if dirn == :L
        f = (v::AbstractVector) -> vec(applyTM_l(A, B, copyto!(x, v)))
    else
        f = (v::AbstractVector) -> vec(applyTM_r(A, B, copyto!(x, v)))
    end
    
    fmap = LinearMap{T}(f, D*DB)
    
    ev, eV, info = eigsolve(f, vec(x0), nev, :LM)
    info.converged < nev && @warn "$(nev - info.converged) eigenvectors did not converge after $(info.numiter) iterations."

    dirn == :L && conj!(ev)
    
    #Reshape eigenvectors to (D,DB) matrices
    eVm = Matrix[reshape(eV[j], (D,DB)) for j in 1:length(eV)]
    
    ev, eVm
end

function tm_eigs_dense(A::MPSTensor{T}, B::MPSTensor{T}) where {T}
    TM = TM_dense(A, B)
    
    DA = bond_dim(A)
    DB = bond_dim(B)
    
    TM = reshape(TM, (DA*DB,DA*DB))
    ev, eVr = eigen(TM)
    eVl = inv(eVr)'
    
    eVmr = Matrix[reshape(eVr[:,j], (DA,DB)) for j in 1:size(eVr, 2)]
    eVml = Matrix[reshape(eVl[:,j], (DA,DB)) for j in 1:size(eVl, 2)]
    
    ev, eVml, eVmr
end

function tm_eigs(A::MPSTensor{T}, B::MPSTensor{T}, min_nev::Integer; D_dense_max::Integer=8) where {T}
    if bond_dim(A) * bond_dim(B) > D_dense_max^2
        evl, eVl = tm_eigs_sparse(A, B, :L, min_nev)
        evr, eVr = tm_eigs_sparse(A, B, :R, min_nev)
        #rely on tm_eigs returning eigenvalues in order of descending magnitude
    else
        evr, eVl, eVr = tm_eigs_dense(A, B)
        evl = evr
    end
    
    evl, evr, eVl, eVr
end

function tm_dominant_eigs(A::MPSTensor{T}, B::MPSTensor{T}; D_dense_max::Integer=8) where {T}
    evl, evr, eVl, eVr = tm_eigs(A, B, 1, D_dense_max=D_dense_max)
    
    indmaxl = argmax(abs.(evl))
    indmaxr = argmax(abs.(evr))
    
    tol = sqrt(eps(real(T))) #If one of the checks hits this upper bound, something went horribly wrong!
    
    if abs(evl[indmaxl] - evr[indmaxr]) > tol
        @warn "Largest magnitude eigenvalues do not match!" evl[indmaxl] evr[indmaxr]
    end
    
    dominant_ev = evl[indmaxl]
    
    if imag(dominant_ev) > tol
        @warn "Largest eigenvalue is not real!" dominant_ev
    end
    
    l = eVl[indmaxl]
    rmul!(l, 1.0/norm(l)) #tm_eigs_dense does not normalize the left eigenvectors
    r = eVr[indmaxr]
    
    #We need this to be 1
    n = vec(l) ⋅ vec(r)
    abs_n = abs(n)
    
    abs_n < tol && @warn "l and r are orthogonal!" abs_n
    
    phase_n = abs_n/n
    sfac = 1.0/sqrt(abs_n)
    
    #We can always find Hermitian l and r. Fix the phase on r using its trace, which must be nonzero.
    r_tr = tr(r)
    phase_r = abs(r_tr)/r_tr
    
    #Fix the phase on l using the phase of n
    phase_l = phase_r * conj(phase_n)
    
    rmul!(l, phase_l * sfac)
    rmul!(r, phase_r * sfac)
    
    #force exact Hermiticity
    lh = (l + l')/2
    rh = (r + r')/2
    
    lh = T <: Real ? real(lh) : lh
    rh = T <: Real ? real(rh) : rh
    
    l = convert(Matrix{T}, lh)
    r = convert(Matrix{T}, rh)
    
    norm(lh - l) > tol && @warn "l was not made Hermitian"
    norm(rh - r) > tol && @warn "r was not made Hermitian"
    
    dominant_ev, l, r
end

function gauge_transform(A::MPSTensor, g::Matrix, gi::Matrix)
    GC.enable(false)
    @tensor Anew[a,s,d] := g[a,b] * A[b,s,c] * gi[c,d] #NOTE: Requires a temporary array
    GC.enable(true)
    GC.gc(false)
    Anew
end

function canonicalize_left(l::Matrix{T}, r::Matrix{T}) where {T}
    tol = sqrt(eps(real(T)))
    
    evl, Ul = eigen(Hermitian(l))
    norm(Ul * Ul' - I) > tol && @warn "Nonunintary eigenvectors."

    sevl = Diagonal(sqrt.(complex.(evl)))
    g = sevl * Ul'
    gi = Ul * inv(sevl)
    
    r = g * Hermitian(r) * g'
    r = Hermitian((r + r')/2)
    
    evr, Ur = eigen(r)
    norm(Ur * Ur' - I) > tol && @warn "Nonunintary eigenvectors."
    
    gi = gi * Ur
    g = Ur' * g
    
    #It would be nice to use special matrix types here, but currently our tm functions can't handle them
    rnew = Diagonal(convert(Vector{T}, evr))
    lnew = I
    
    lnew, rnew, g, gi
end


MPO_open{T} = Vector{MPOTensor{T}}

#Multiplies MPO tensors top to bottom: A*B connects the bottom index of A with the top index of B.
function mul_MPO(A::MPOTensor, B::MPOTensor)
    M1l,d1t,M1r,d1b = size(A)
    M2l,d2t,M2r,d2b = size(B)
    @tensor AB[m1l,m2l,pt,m1r,m2r,pb] := A[m1l,pt,m1r,p] * B[m2l,p,m2r,pb]
    reshape(AB, (M1l*M2l, d1t, M1r*M2r, d2b))
end

function mul_MPO(M1::NTuple{N,MPOTensor{T}}, M2::NTuple{N,MPOTensor{T}}) where {N,T}
    ntuple(j -> mul_MPO(M1[j], M2[j]), N)
end

function mul_MPO(M1::Vector{MPOTensor{T}}, M2::Vector{MPOTensor{T}}) where {T}
    MPOTensor{T}[mul_MPO(M1[j], M2[j]) for j in 1:length(M1)]
end


function TM_dense_MPO(A::MPSTensor, B::MPSTensor, o::MPOTensor)::MPS_MPO_TM
    GC.enable(false)
    @tensor TM[k1,m1,b1, k2,m2,b2] := (A[k1, s1, k2] * o[m1,s1,m2,t1]) * conj(B[b1, t1, b2])
    GC.enable(true)
    GC.gc(false)
    TM
end

function applyTM_MPO_l(A::MPSTensor, B::MPSTensor, o::MPOTensor, TM2::MPS_MPO_TM)::MPS_MPO_TM
    GC.enable(false)
    @tensor TM21[k1,m1,b1, k3,m3,b3] := ((TM2[k1,m1,b1, k2,m2,b2] * conj(B[b2, t2, b3])) * o[m2,s2,m3,t2]) * A[k2, s2, k3]
    GC.enable(true)
    GC.gc(false)
    TM21
end

function applyTM_MPO_r(A::MPSTensor, B::MPSTensor, o::MPOTensor, TM2::MPS_MPO_TM)::MPS_MPO_TM
    GC.enable(false)
    @tensor TM12[k1,m1,b1, k3,m3,b3] :=  conj(B[b1, t1, b2]) * (o[m1,s1,m2,t1] * (A[k1, s1, k2] * TM2[k2,m2,b2, k3,m3,b3]))
    GC.enable(true)
    GC.gc(false)
    TM12
end

function worklen_applyTM_MPO_l(A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    len = 0
    
    wsz1 = (size(TM2)[1:5]..., size(B1)[2:3]...)
    wlen1 = prod(wsz1)
    wsz2 = (wsz1[1:4]..., wsz1[7], wsz1[5:6]...)
    wlen2 = prod(wsz2)
    
    len = max(len, wlen2 + wlen1)

    opsz = size(o)
    wsz3 = (opsz[1], opsz[4], opsz[2:3]...)
    wlen3 = prod(wsz3)
    
    wsz4 = (wsz2[1:5]..., wsz3[3:4]...)
    wlen4 = prod(wsz4)
    
    len = max(len, wlen1 + wlen3 + wlen4)
    
    wsz5 = (wsz4[1:3]..., wsz4[5], wsz4[7], wsz4[4], wsz4[6])
    wlen5 = prod(wsz5)
    
    wsz6 = (wsz5[1:5]..., size(A1)[3])
    wlen6 = prod(wsz6)
    
    if wlen5 <= wlen1+wlen3
        ind5 = 1
    else
        ind5 = wlen1+wlen3+wlen4+1
    end
    
    len = max(len, ind5-1 + wlen5)
    
    if ind5 == 1 || wlen6 >= ind5
        len = max(len, ind5-1 + wlen5 + wlen6)
    else
        len = max(len, wlen6)
    end

    len
end

function workvec_applyTM_MPO_l(A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    Vector{T}(undef, worklen_applyTM_MPO_l(A1, B1, o, TM2))
end

function workvec_applyTM_MPO_l!(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    len = worklen_applyTM_MPO_l(A1, B1, o, TM2)
    work = length(work) < len ? resize!(work, len) : work
end

function res_applyTM_MPO_l(A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})::MPS_MPO_TM{T} where {T}
    Array{T,6}(undef, (size(TM2)[1:3]..., bond_dim_R(A1),size(o,3),bond_dim_R(B1)))
end

function res_applyTM_MPO_l!(res::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})::MPS_MPO_TM{T} where {T}
    sz = (size(TM2)[1:3]..., bond_dim_R(A1), size(o,3), bond_dim_R(B1))
    size(res) == sz ? res : zeros(T, sz)
end

#NOTE: This works, but is pretty horrible. We are doing manual memory management because the garbage collector is super slow!
function applyTM_MPO_l!(TM21::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T} where {T}
    wsz1 = (size(TM2)[1:5]..., size(B1)[2:3]...)
    wlen1 = prod(wsz1)
    wsz2 = (wsz1[1:4]..., wsz1[7], wsz1[5:6]...)
    wlen2 = prod(wsz2)
    
    TM2B1 = unsafe_reshaped_subvec(work, wlen2+1, wsz1)
    @tensor TM2B1[k1,m1,b1, k2,m2,t2,b3] = TM2[k1,m1,b1, k2,m2,b2] * conj(B1[b2, t2, b3])
    
    TM2B1p = unsafe_reshaped_subvec(work, 1, wsz2)
    @tensor TM2B1p[k1,m1,b1, k2,b3, m2,t2] = TM2B1[k1,m1,b1, k2,m2,t2,b3]
    
    opsz = size(o)
    wsz3 = (opsz[1], opsz[4], opsz[2:3]...)
    wlen3 = prod(wsz3)
    
    wsz4 = (wsz2[1:5]..., wsz3[3:4]...)
    wlen4 = prod(wsz4)
    
    op = unsafe_reshaped_subvec(work, wlen1+1, wsz3)
    @tensor op[m2,t2,s2,m3] = o[m2,s2,m3,t2]
    
    TM2B1pop = unsafe_reshaped_subvec(work, wlen1+wlen3+1, wsz4)
    @tensor TM2B1pop[k1,m1,b1, k2,b3, s2,m3] = TM2B1p[k1,m1,b1, k2,b3, m2,t2] * op[m2,t2,s2,m3]
    
    wsz5 = (wsz4[1:3]..., wsz4[5], wsz4[7], wsz4[4], wsz4[6])
    wlen5 = prod(wsz5)
    
    wsz6 = (wsz5[1:5]..., size(A1)[3])
    wlen6 = prod(wsz6)
    
    if wlen5 <= wlen1+wlen3
        ind5 = 1
    else
        ind5 = wlen1+wlen3+wlen4+1
    end
    TM2B1pop_p = unsafe_reshaped_subvec(work, ind5, wsz5)
    @tensor TM2B1pop_p[k1,m1,b1, b3, m3, k2,s2] = TM2B1pop[k1,m1,b1, k2,b3, s2,m3]
    
    if ind5 == 1 || wlen6 >= ind5
        TM21p = unsafe_reshaped_subvec(work, wlen5+ind5, wsz6)
    else
        TM21p = unsafe_reshaped_subvec(work, 1, wsz6)
    end
    @tensor TM21p[k1,m1,b1, b3,m3,k3] = TM2B1pop_p[k1,m1,b1, b3,m3, k2,s2] * A1[k2,s2, k3]
    
    @tensor TM21[k1,m1,b1, k3,m3,b3] = TM21p[k1,m1,b1, b3,m3,k3]
    TM21
end

#Convenience functions that fall back on MPO-less versions in case o is the Identity.
workvec_applyTM_MPO_l(A1::MPSTensor{T}, B1::MPSTensor{T}, o::IdentityMPOTensor, TM2::MPS_MPO_TM{T})  where {T} = workvec_applyTM_l(A1, B1, TM2)
workvec_applyTM_MPO_l!(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::IdentityMPOTensor, TM2::MPS_MPO_TM{T}) where {T} = workvec_applyTM_l!(work, A1, B1, TM2)
function applyTM_MPO_l!(TM21::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::IdentityMPOTensor, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T} where {T}
    applyTM_l!(TM21, A1, B1, TM2, work)
end

function workvec_alloc_applyTM_MPO_r(A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    ATM2_sz = (size(A)[1:2]..., size(TM2)[2:end]...)
    ATM2_len = prod(ATM2_sz)
    
    ATM2p_sz = (ATM2_sz[2:3]..., ATM2_sz[1], ATM2_sz[4:end]...)
    ATM2p_len = prod(ATM2p_sz)
    
    op_sz = (size(o)[1], size(o)[4], size(o)[2], size(o)[3])
    op_len = prod(op_sz)
    
    oATM2_sz = (op_sz[1:2]..., ATM2p_sz[3:end]...)
    oATM2_len = prod(oATM2_sz)
    
    oATM2p_sz = (oATM2_sz[2], oATM2_sz[4], oATM2_sz[1], oATM2_sz[3], oATM2_sz[5:end]...)
    oATM2p_len = prod(oATM2p_sz)
    
    TM21p_sz = (size(B)[1], oATM2p_sz[3:end]...)
    TM21p_len = prod(TM21p_sz)
    
    ATM2_ind = ATM2p_len+1
    ATM2p_ind = 1
    op_ind = ATM2p_len+1
    oATM2_ind = ATM2p_len+op_len+1
    if ATM2p_len+op_len >= oATM2_len
        oATM2p_ind = 1
    else
        oATM2p_ind = ATM2p_len+op_len+oATM2_len+1
    end
    if oATM2p_ind >= TM21p_len
        TM21p_ind = 1
    else
        TM21p_ind = oATM2p_ind + oATM2p_len
    end
    
    (ATM2_sz, ATM2p_sz, op_sz, oATM2_sz, oATM2p_sz, TM21p_sz), (ATM2_ind, ATM2p_ind, op_ind, oATM2_ind, oATM2p_ind, TM21p_ind)
end

function workvec_applyTM_MPO_r!(work::Vector{T}, A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) where {T}
    szs, inds = workvec_alloc_applyTM_MPO_r(A, B, o, TM2)
    
    len = maximum(inds[j] + prod(szs[j]) for j in 1:6)
    
    work = length(work) < len ? resize!(work, len) : work
end

workvec_applyTM_MPO_r(A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) where {T} = 
    workvec_applyTM_MPO_r!(Vector{T}(), A, B, o, TM2)

function res_applyTM_MPO_r(A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})::MPS_MPO_TM{T} where {T}
    Array{T,6}(undef, (bond_dim_L(A),size(o,1),bond_dim_L(B),size(TM2)[4:6]...))
end

function res_applyTM_MPO_r!(res::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})::MPS_MPO_TM{T} where {T}
    sz = ((bond_dim_L(A),size(o,1),bond_dim_L(B),size(TM2)[4:6]...))
    size(res) == sz ? res : zeros(T, sz)
end

function applyTM_MPO_r!(TM21::MPS_MPO_TM{T}, A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T} where {T}
    szs, inds = workvec_alloc_applyTM_MPO_r(A, B, o, TM2)
    ATM2_sz, ATM2p_sz, op_sz, oATM2_sz, oATM2p_sz, TM21p_sz = szs
    ATM2_ind, ATM2p_ind, op_ind, oATM2_ind, oATM2p_ind, TM21p_ind = inds
    
    ATM2 = unsafe_reshaped_subvec(work, ATM2_ind, ATM2_sz)
    @tensor ATM2[k1,s1, m2,b2, k3,m3,b3] = A[k1, s1, k2] * TM2[k2,m2,b2, k3,m3,b3]
    
    ATM2p = unsafe_reshaped_subvec(work, ATM2p_ind, ATM2p_sz)
    @tensor ATM2p[s1,m2, k1, b2, k3,m3,b3] = ATM2[k1,s1, m2,b2, k3,m3,b3]
    
    op = unsafe_reshaped_subvec(work, op_ind, op_sz)
    @tensor op[m1,t1,s1,m2] = o[m1,s1,m2,t1]
    
    oATM2 = unsafe_reshaped_subvec(work, oATM2_ind, oATM2_sz)
    @tensor oATM2[m1,t1, k1, b2, k3,m3,b3] = op[m1,t1,s1,m2] * ATM2p[s1,m2, k1, b2, k3,m3,b3]
    
    oATM2p = unsafe_reshaped_subvec(work, oATM2p_ind, oATM2p_sz)
    @tensor oATM2p[t1,b2, m1, k1, k3,m3,b3] = oATM2[m1,t1, k1, b2, k3,m3,b3]
    
    TM21p = unsafe_reshaped_subvec(work, TM21p_ind, TM21p_sz)
    @tensor TM21p[b1, m1, k1, k3,m3,b3] = conj(B[b1, t1, b2]) * oATM2p[t1,b2, m1, k1, k3,m3,b3]
    
    @tensor TM21[k1,m1,b1, k3,m3,b3] = TM21p[b1, m1, k1, k3,m3,b3]
end

workvec_applyTM_MPO_r(A1::MPSTensor{T}, B1::MPSTensor{T}, o::IdentityMPOTensor, TM2::MPS_MPO_TM{T}) where {T} = workvec_applyTM_r(A1, B1, TM2)
workvec_applyTM_MPO_r!(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::IdentityMPOTensor, TM2::MPS_MPO_TM{T}) where {T} = workvec_applyTM_r!(work, A1, B1, TM2)
function applyTM_MPO_r!(TM21::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::IdentityMPOTensor, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T} where {T}
    applyTM_r!(TM21, A1, B1, TM2, work)
end

end