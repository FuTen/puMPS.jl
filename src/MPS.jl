
module MPS

using PyCall
using TensorOperations
using NCon
using LinearMaps

export MPSTensor, bond_dim, phys_dim, mps_tensor, mps_tensor_shape, rand_MPSTensor_unitary, rand_MPSTensor,
       MPS_TM, TM_dense, TM_dense_op_nn, 
       applyTM_op_nn_l, applyTM_l, workvec_applyTM_l, workvec_applyTM_l!, applyTM_l!, applyTM_r, 
       tm_eigs_sparse, tm_eigs_dense, tm_dominant_eigs,
        gauge_transform, canonicalize_left,
       MPOTensor, MPO_open, MPS_MPO_TM, TM_dense_MPO, applyTM_MPO_l, applyTM_MPO_r,
       worklen_applyTM_MPO_l, workvec_applyTM_MPO_l, workvec_applyTM_MPO_l!, res_applyTM_MPO_l, applyTM_MPO_l!,
       workvec_applyTM_MPO_r, workvec_applyTM_MPO_r!, applyTM_MPO_r!, res_applyTM_MPO_r,
       TM_convert

"""
We will use the convention that the middle index is the physical one,
while the first and third are the left and right virtual indices.
In pictures, for an `MPSTensor` `A`:

 1--A--3
    |
    2

"""
typealias MPSTensor{T} Array{T,3}

bond_dim(A::MPSTensor, which::Int=1) = which == 1 ? size(A,1) : size(A,3)
phys_dim(A::MPSTensor) = size(A,2)
mps_tensor_shape(d::Int, D::Int) = (D,d,D)
mps_tensor_shape(d::Int, D1::Int, D2::Int) = (D1,d,D2)

function randunitary{T}(::Type{T}, N::Int)::Matrix{T}
    if T <: Complex
        rT = real(T)
        A = complex(randn(rT, N,N), randn(rT, N,N)) / √2
    else
        A = randn(T, N,N)
    end
    Q, R = qr(A)
    r = diag(R)
    L = diagm(r ./ abs(r))
    Q*L
end

#Note: An MPS generated this way already has r proportional to I and largest tm eigenvalue = 1
function rand_MPSTensor_unitary{T}(::Type{T}, d::Int, D::Int)::MPSTensor{T}
    U = randunitary(T, d*D)
    reshape(U[1:D,:], mps_tensor_shape(d,D))
end

function rand_MPSTensor{T}(::Type{T}, d::Int, D::Int)::MPSTensor{T}
    shp = mps_tensor_shape(d,D)
    if T <: Complex
        rT = real(T)
        A = complex(randn(rT, shp), randn(rT, shp)) / √2
    else
        A = randn(T, shp)
    end
    A
end


"""
Dense transfer matrix:

 1----A-----3
      |
 2--conj(B)-4

"""
typealias MPS_TM{T} Array{T,4} 

Base.trace{T}(TM::MPS_TM{T}) = ncon(TM, (1,2,1,2))[1]

function TM_dense(A::MPSTensor, B::MPSTensor)::MPS_TM
    @tensor TM[lA,lB,rA,rB] := A[lA, s, rA] * conj(B[lB, s, rB]) #NOTE: This will generally allocate to do permutations!
    TM
end

function TM_dense_op_nn{T}(A1::MPSTensor, A2::MPSTensor, B1::MPSTensor, B2::MPSTensor, op::Array{T,4})::MPS_TM
    @tensor TM[lA1,lB1,rA2,rB2] := ((A1[lA1, p1k, iA] * A2[iA, p2k, rA2]) * op[p1b, p2b, p1k, p2k]) * (conj(B1[lB1, p1b, iB]) * conj(B2[iB, p2b, rB2])) #NOTE: Allocates intermediate arrays
    TM
end

function applyTM_op_nn_l{T}(A1::MPSTensor, A2::MPSTensor, B1::MPSTensor, B2::MPSTensor, op::Array{T,4}, TM2::MPS_TM{T})
    @tensor TM21[k1,b1,k4,b4] := ((((TM2[k1,b1,k2,b2] * A1[k2,s1,k3]) * op[t1,t2,s1,s2]) * conj(B1[b2,t1,b3])) * A2[k3,s2,k4]) * conj(B2[b3,t2,b4])
    TM21
end

#This is a workaround for StridedArray not including reshaped arrays formed from a FastContiguousSubArray in in Julia <= 0.6.
#Since TensorOperations uses the StridedArray type, it misses these cases. Here, we use pointer manipulation to generate
#views of sections of a vector with an arbitrary "size".
function unsafe_reshaped_subvec{T}(w::Vector{T}, ind1::Int, sz::NTuple)
    len = prod(sz)
    @assert length(w) >= ind1+len-1 "Vector is not big enough to hold subarray!"
    unsafe_wrap(Array, pointer(w, ind1), sz)
end

workvec_applyTM_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}) = Vector{T}(bond_dim(A1)^2 * bond_dim(B1)^2 * phys_dim(A1) * 2)

function worklen_applyTM_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_TM{T})
    size(TM2,1) * size(TM2,2) * size(B1,3) * size(A1,3) * phys_dim(A1) * 2
end

function workvec_applyTM_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_TM{T})
    Vector{T}(worklen_applyTM_l(A1, B1, TM2))
end

function workvec_applyTM_l!{T}(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_TM{T})
    len = worklen_applyTM_l(A1, B1, TM2)
    work = length(work) < len ? resize!(work, len) : work
end

function applyTM_l!{T}(TM21::MPS_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_TM{T}, work::Vector{T})
    wsz1 = (size(TM2)[1:3]..., size(B1)[2:3]...)
    wlen1 = prod(wsz1)
    TM2B1 = unsafe_reshaped_subvec(work, 1, wsz1)
    @tensor TM2B1[lA,lB, mA,s,rB] = TM2[lA,lB, mA,mB] * conj(B1[mB,s,rB])
    
    wsz2 = (wsz1[1:2]..., wsz1[5], wsz1[3:4]...)
    wlen2 = prod(wsz2)
    TM2B1p = unsafe_reshaped_subvec(work, wlen1+1, wsz2) #take work vector section after TM2B1 storage
    @tensor TM2B1p[lA,lB, rB,mA,s] = TM2B1[lA,lB, mA,s,rB]
    
    TM2sz = size(TM2)
    wsz3 = (TM2sz[1:2]..., TM2sz[4], TM2sz[3])
    @assert prod(wsz3) <= wlen1
    #Assuming this array must be <= length of TM2B1, we can reuse that memory here. (May not work with nonuniform D).
    TM21p = unsafe_reshaped_subvec(work, 1, wsz3)
    @tensor TM21p[lA,lB, rB,rA] = TM2B1p[lA,lB, rB,mA,s] * A1[mA,s,rA]
    @tensor TM21[lA,lB, rA,rB] = TM21p[lA,lB, rB,rA]
    TM21
end

function applyTM_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}, TM2::MPS_TM{T})
    @tensor TM21[lA,lB, rA,rB] := (TM2[lA,lB, mA,mB] * conj(B1[mB,s,rB])) * A1[mA,s,rA] #NOTE: Intermediates, final permutation.
end

function applyTM_l(A::MPSTensor, B::MPSTensor, x::Matrix)
    @tensor res[a, d] := conj(A[b, s, a]) * (x[b, c] * B[c, s, d]) #NOTE: This requires at least one intermediate array
end
applyTM_l(A::MPSTensor, B::MPSTensor, x::Vector) = applyTM_l(A, B, reshape(x, (bond_dim(A), bond_dim(B))))

function applyTM_r(A::MPSTensor, B::MPSTensor, x::Matrix)
    @tensor res[a, d] := (A[a, s, b] * x[b, c]) * conj(B[d, s, c]) #NOTE: This requires at least one intermediate array
end
applyTM_r(A::MPSTensor, B::MPSTensor, x::Vector) = applyTM_r(A, B, reshape(x, (bond_dim(A), bond_dim(B))))

function tm_eigs_sparse{T}(A::MPSTensor{T}, B::MPSTensor{T}, dirn::Symbol, nev::Int=1; 
                    x0::Matrix{T}=ones(T,(bond_dim(A), bond_dim(B))))
    @assert dirn in (:L, :R)
    
    D = bond_dim(A)
    DB = bond_dim(B)
    
    x = zeros(T, (D,DB))
    if dirn == :L
        f = (v::AbstractVector) -> vec(applyTM_l(A, B, copy!(x, v)))
    else
        f = (v::AbstractVector) -> vec(applyTM_r(A, B, copy!(x, v)))
    end
    
    fmap = LinearMap(f, D*DB, T)
    
    ev, eV, nconv, niter, nmult, resid = eigs(fmap, nev=nev, which=:LM, ritzvec=true, v0=vec(x0))
    
    #Reshape eigenvectors to (D,DB) matrices
    eVm = Matrix[reshape(eV[:,j], (D,DB)) for j in 1:size(eV, 2)]
    
    ev, eVm
end

function tm_eigs_dense{T}(A::MPSTensor{T}, B::MPSTensor{T})
    TM = TM_dense(A, B)
    
    DA = bond_dim(A)
    DB = bond_dim(B)
    
    TM = reshape(TM, (DA*DB,DA*DB))
    ev, eVr = eig(TM)
    eVl = inv(eVr)'
    
    eVmr = Matrix[reshape(eVr[:,j], (DA,DB)) for j in 1:size(eVr, 2)]
    eVml = Matrix[reshape(eVl[:,j], (DA,DB)) for j in 1:size(eVl, 2)]
    
    ev, eVml, eVmr
end

function tm_eigs{T}(A::MPSTensor{T}, B::MPSTensor{T}, min_nev::Int; D_dense_max::Int=8)
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

function tm_dominant_eigs{T}(A::MPSTensor{T}, B::MPSTensor{T}; D_dense_max::Int=8)
    evl, evr, eVl, eVr = tm_eigs(A, B, 1, D_dense_max=D_dense_max)
    
    indmaxl = findmax(abs(evl))[2]
    indmaxr = findmax(abs(evr))[2]
    
    tol = sqrt(eps(real(T))) #If one of the checks hits this upper bound, something went horribly wrong!
    
    if abs(evl[indmaxl] - evr[indmaxr]) > tol
        warn("Largest magnitude eigenvalues do not match!", evl[indmaxl], evr[indmaxr])
    end
    
    dominant_ev = evl[indmaxl]
    
    if imag(dominant_ev) > tol
        warn("Largest eigenvalue is not real!", dominant_ev)
    end
    
    l = eVl[indmaxl]
    scale!(l, 1.0/vecnorm(l)) #tm_eigs_dense does not normalize the left eigenvectors
    r = eVr[indmaxr]
    
    #We need this to be 1
    n = vec(l) ⋅ vec(r)
    abs_n = abs(n)
    
    abs_n < tol && warn("l and r are orthogonal!", abs_n)
    
    phase_n = abs_n/n
    sfac = 1.0/sqrt(abs_n)
    
    #We can always find Hermitian l and r. Fix the phase on r using its trace, which must be nonzero.
    r_tr = trace(r)
    phase_r = abs(r_tr)/r_tr
    
    #Fix the phase on l using the phase of n
    phase_l = phase_r * conj(phase_n)
    
    scale!(l, phase_l * sfac)
    scale!(r, phase_r * sfac)
    
    #force exact Hermiticity
    lh = 0.5(l + l')
    rh = 0.5(r + r')
    
    lh = T <: Real ? real(lh) : lh
    rh = T <: Real ? real(rh) : rh
    
    l = convert(Matrix{T}, lh)
    r = convert(Matrix{T}, rh)
    
    vecnorm(lh - l) > tol && warn("l was not made Hermitian")
    vecnorm(rh - r) > tol && warn("r was not made Hermitian")
    
    dominant_ev, l, r
end

function gauge_transform(A::MPSTensor, g::Matrix, gi::Matrix)
    @tensor Anew[a,s,d] := g[a,b] * A[b,s,c] * gi[c,d] #NOTE: Requires a temporary array
end

function canonicalize_left{T}(l::Matrix{T}, r::Matrix{T})
    tol = sqrt(eps(real(T)))
    
    evl, Ul = eig(Hermitian(l))
    vecnorm(Ul * Ul' - I) > tol && warn("Nonunintary eigenvectors.")

    sevl = Diagonal(sqrt(complex(evl)))
    g = sevl * Ul'
    gi = Ul * inv(sevl)
    
    r = g * Hermitian(r) * g'
    r = Hermitian(0.5(r + r'))
    
    evr, Ur = eig(r)
    vecnorm(Ur * Ur' - I) > tol && warn("Nonunintary eigenvectors.")
    
    gi = gi * Ur
    g = Ur' * g
    
    #It would be nice to use special matrix types here, but currently our tm functions can't handle them
    rnew = Diagonal(convert(Vector{T}, evr))
    lnew = I
    
    lnew, rnew, g, gi
end


"""
MPO tensor:

     2
     |
 1---o---3
     |
     4
"""
typealias MPOTensor{T} Array{T,4}

typealias MPO_open{T} Vector{MPOTensor{T}}

"""
Dense transfer matrix with MPO:

 1----A-----4
      |
 2----o-----5
      |
 3--conj(B)-6
"""
typealias MPS_MPO_TM{T} Array{T,6}

Base.trace{T}(TM::MPS_MPO_TM{T}) = ncon(TM, (1,2,3,1,2,3))[1]


#Turn the MPS TM into an MPS TM with (size 1) MPO indices
TM_convert{T}(TM::MPS_TM{T}) = reshape(TM, (size(TM,1),1,size(TM,2), size(TM,3),1,size(TM,4)))::MPS_MPO_TM{T}

function TM_convert{T}(TM::MPS_MPO_TM{T})::MPS_TM{T}
    size(TM,2) == size(TM,5) == 1 || error("MPO bond dimensions not equal to 1!")
    squeeze(TM, (2,5))
end

function TM_dense_MPO(A::MPSTensor, B::MPSTensor, o::MPOTensor)::MPS_MPO_TM
    @tensor TM[k1,m1,b1, k2,m2,b2] := (A[k1, s1, k2] * o[m1,s1,m2,t1]) * conj(B[b1, t1, b2])
end

function applyTM_MPO_l(A::MPSTensor, B::MPSTensor, o::MPOTensor, TM2::MPS_MPO_TM)::MPS_MPO_TM
    @tensor TM21[k1,m1,b1, k3,m3,b3] := ((TM2[k1,m1,b1, k2,m2,b2] * conj(B[b2, t2, b3])) * o[m2,s2,m3,t2]) * A[k2, s2, k3]
end

function applyTM_MPO_r(A::MPSTensor, B::MPSTensor, o::MPOTensor, TM2::MPS_MPO_TM)::MPS_MPO_TM
    @tensor TM12[k1,m1,b1, k3,m3,b3] :=  conj(B[b1, t1, b2]) * (o[m1,s1,m2,t1] * (A[k1, s1, k2] * TM2[k2,m2,b2, k3,m3,b3]))
end

function worklen_applyTM_MPO_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})
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

function workvec_applyTM_MPO_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})
    Vector{T}(worklen_applyTM_MPO_l(A1, B1, o, TM2))
end

function workvec_applyTM_MPO_l!{T}(work::Vector{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})
    len = worklen_applyTM_MPO_l(A1, B1, o, TM2)
    work = length(work) < len ? resize!(work, len) : work
end

function res_applyTM_MPO_l{T}(A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})::MPS_MPO_TM{T}
    Array{T,6}((size(TM2)[1:3]..., bond_dim(A1,2),size(o,3),bond_dim(B1,2)))
end

#NOTE: This works, but is pretty horrible. We are doing manual memory management because the garbage collector is super slow!
function applyTM_MPO_l!{T}(TM21::MPS_MPO_TM{T}, A1::MPSTensor{T}, B1::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T}
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

function workvec_alloc_applyTM_MPO_r{T}(A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})
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

function workvec_applyTM_MPO_r!{T}(work::Vector{T}, A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})
    szs, inds = workvec_alloc_applyTM_MPO_r(A, B, o, TM2)
    
    len = maximum(inds[j] + prod(szs[j]) for j in 1:6)
    
    work = length(work) < len ? resize!(work, len) : work
end

workvec_applyTM_MPO_r{T}(A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}) = workvec_applyTM_MPO_r!(Vector{T}(), A, B, o, TM2)

function res_applyTM_MPO_r{T}(A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T})::MPS_MPO_TM{T}
    Array{T,6}((bond_dim(A,1),size(o,1),bond_dim(B,1),size(TM2)[4:6]...))
end

function applyTM_MPO_r!{T}(TM21::MPS_MPO_TM{T}, A::MPSTensor{T}, B::MPSTensor{T}, o::MPOTensor{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T}
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


end