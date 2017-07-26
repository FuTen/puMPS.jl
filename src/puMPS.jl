
module puMPS

using PyCall
using TensorOperations
using NCon
using LinearMaps
using Optim

using MPS

export puMPState, rand_puMPState, mps_tensor, num_sites, set_mps_tensor!,
       apply_blockTM_l, blockTM_dense,
       expect_nn, expect,
       canonicalize_left!,
       minimize_energy_local!,
       excitations!, excitations, tangent_space_metric_and_hamiltonian, 
       tangent_space_Hn, Hn_in_basis,
       ising_local_MPO, ising_PBC_MPO, ising_PBC_MPO_split, ising_OBC_MPO, ising_Hn_MPO_split,
       ANNNI_local_MPO, ANNNI_OBC_MPO, ANNNI_PBC_MPO_split,
       potts3_local_MPO, potts3_OBC_MPO, potts3_PBC_MPO_split

"""
    typealias MPO_PBC_uniform{T} Tuple{MPOTensor{T},MPOTensor{T}}

One middle (bulk) tensor, one boundary tensor.
"""
typealias MPO_PBC_uniform{T} Tuple{MPOTensor{T},MPOTensor{T}}

"""
    typealias MPO_open_uniform{T} Tuple{MPOTensor{T},MPOTensor{T},MPOTensor{T}} 

The left, middle (bulk), and right tensors of an OBC Hamiltonian.
"""
typealias MPO_open_uniform{T} Tuple{MPOTensor{T},MPOTensor{T},MPOTensor{T}} 

"""
    typealias MPO_PBC_split{T} Tuple{MPO_open_uniform{T}, MPO_open{T}}

An OBC Hamiltonian and a further open MPO describing the boundary terms. The sum is a circle Hamiltonian.
"""
typealias MPO_PBC_split{T} Tuple{MPO_open_uniform{T}, MPO_open{T}}


"""
    A uniform Matrix Product State with periodic boundary conditions. 
    We need only store the MPS tensor `A`, which is the same for every site,
    and the number of sites for which `A` is intended to be used.
"""
type puMPState{T}
    A::MPSTensor{T}
    N::Int #number of sites
end

#Generates a random `puMPState` in left-canonical form
rand_puMPState{T}(::Type{T}, d::Int, D::Int, N::Int) = puMPState(rand_MPSTensor_unitary(T, d, D), N)::puMPState{T}

Base.copy(M::puMPState) = puMPState(copy(M.A), M.N)

MPS.bond_dim(M::puMPState) = bond_dim(M.A)
MPS.phys_dim(M::puMPState) = phys_dim(M.A)
mps_tensor(M::puMPState) = M.A
num_sites(M::puMPState) = M.N

set_mps_tensor!{T}(M::puMPState{T}, A::MPSTensor{T}) = M.A = A

"""
    canonicalize_left!(M::puMPState; pinv_tol::Float64=1e-12)

Modifies a puMPState in place via a gauge transformation to bring it into left-canonical form,
returning the puMPState and the gauge-transformation matrices.
"""
function canonicalize_left!(M::puMPState; pinv_tol::Float64=1e-12)
    A = mps_tensor(M)
    
    dominant_ev, l, r = tm_dominant_eigs(A, A)
    
    lnew, rnew, x, xi = MPS.canonicalize_left(l, r)

    AL = gauge_transform(A, x, xi)
    set_mps_tensor!(M, AL)
    
    lambda = Diagonal(sqrt(diag(rnew)))
    
    lambda_i = pinv(lambda, pinv_tol)
    
    M, lambda, lambda_i
end

MPS.canonicalize_left(M::puMPState) = canonicalize_left!(copy(M))

#Computes the one-site transfer matrix of a `puMPState` and returns it as an `MPS_TM` (see MPS).
MPS.TM_dense(M::puMPState) = TM_dense(mps_tensor(M), mps_tensor(M))

"""
    apply_blockTM_l{T}(M::puMPState{T}, TM::MPS_TM{T}, N::Int)

Applies `N` transfer matrices `TM_M` of the puMPState `M` 
to an existing transfer matrix `TM` by acting to the left:
```
TM * (TM_M)^N
```
"""
function apply_blockTM_l{T}(M::puMPState{T}, TM::MPS_TM{T}, N::Int)
    A = mps_tensor(M)
    work = workvec_applyTM_l(A, A)
    TMres = zeros(TM)
    TM = N > 1 ? copy(TM) : TM #Never overwrite TM!
    for i in 1:N
        applyTM_l!(TMres, A, A, TM, work) #D^5 d
        TM, TMres = (TMres, TM)
    end
    TM
end

"""
    blockTM_dense{T}(M::puMPState{T}, N::Int)

Computes the `N`th power of the transfer matrix of the puMPState `M`. Time cost: O(`bond_dim(M)^5`).
"""
function blockTM_dense{T}(M::puMPState{T}, N::Int)
    #TODO: Depending on d vs. D, block the MPS tensors first to form an initial blockTM at cost D^4 d^blocksize.
    D = bond_dim(M)
    TM = N == 0 ? reshape(kron(eye(T,D),eye(T,D)), (D,D,D,D)) : apply_blockTM_l(M, TM_dense(M), N-1)
    TM
end

"""
    blockTMs{T}(M::puMPState{T}, N::Int=num_sites(M))

Computes the powers of the transfer matrix of the puMPState `M` up to and including the `N`th power.
Time cost: O(`bond_dim(M)^5`).
"""
function blockTMs{T}(M::puMPState{T}, N::Int=num_sites(M))
    A = mps_tensor(M)
    TMs = MPS_TM{T}[TM_dense(M)]
    work = workvec_applyTM_l(A, A)
    for n in 2:N
        TMres = similar(TMs[end])
        push!(TMs, applyTM_l!(TMres, A, A, TMs[end], work))
    end
    TMs
end

"""
    Base.norm{T}(M::puMPState{T}; TM_N::MPS_TM{T}=blockTM_dense(M, num_sites(M)))

The norm of the puMPState `M`, optionally reusing the precomputed 
`N`th power of the transfer matrix `TM_N`, where `N = num_sites(M)`.
"""
function Base.norm{T}(M::puMPState{T}; TM_N::MPS_TM{T}=blockTM_dense(M, num_sites(M)))
    sqrt(trace(TM_N))
end

"""
    Base.normalize!{T}(M::puMPState{T}; TM_N::MPS_TM{T}=blockTM_dense(M, num_sites(M)))
    
Normalizes the puMPState `M` in place, optionally reusing the precomputed 
`N`th power of the transfer matrix `TM_N`, where `N = num_sites(M)`.
"""
function Base.normalize!{T}(M::puMPState{T}; TM_N::MPS_TM{T}=blockTM_dense(M, num_sites(M)))
    scale!(mps_tensor(M), 1.0 / norm(M, TM_N=TM_N)^(1.0/num_sites(M)) )
    M
end
    
Base.normalize{T}(M::puMPState{T}; TM_N::MPS_TM{T}=blockTM_dense(M, num_sites(M))) = normalize!(copy(M), TM_N=TM_N)

"""
    Base.normalize!{T}(M::puMPState{T}, blkTMs::Vector{MPS_TM{T}})

Normalizes the puMPState `M` in place together with a set of precomputed powers of the transfer matrix `blkTMs`.
"""
function Base.normalize!{T}(M::puMPState{T}, blkTMs::Vector{MPS_TM{T}})
    N = num_sites(M)
    normM = norm(M, TM_N=blkTMs[N])
    
    scale!(mps_tensor(M), 1.0 / normM^(1.0/N) )
    for n in 1:N
        scale!(blkTMs[n], 1.0 / normM^(2.0/n))
    end
    M, blkTMs
end

"""
    expect_nn{Ts, Top}(M::puMPState{Ts}, op::Array{Top,4}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_TM{Ts}}=MPS_TM{T}[])

Computes the expectation value with respect to `M` of a nearest-neighbour operator,
supplied as a 4-dimensional `Array` defined as
    `op[t1,t2, s1,s2]` = <t1,t2|op|s1,s2>
with each index enumerating the basis for the one-site physical Hilbert space
according to which the puMPState `M` is defined.

Optionally uses precomputed powers of the transfer matrix as `blkTMs`.
In case `MPS_is_normalized == false` computes the norm of `M` at the same time.
"""
function expect_nn{Ts, Top}(M::puMPState{Ts}, op::Array{Top,4}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_TM{Ts}}=MPS_TM{T}[])
    N = num_sites(M)
    A = mps_tensor(M)
    
    if MPS_is_normalized && length(blkTMs) < N-2
        TM = TM_dense_op_nn(A,A,A,A, op) #D^4 d^2
        TM = apply_blockTM_l(M, TM, N-2) #NOTE: If N-2 >> D it is cheaper to do full D^6 multiplication with a block.
        
        return trace(TM)
    else
        #We can use most of the block TM for both the norm and the expectation value.
        TM = length(blkTMs) >= N-2 ? blkTMs[N-2] : blockTM_dense(M, N-2)
        normsq = length(blkTMs) == N ? trace(blkTMs[N]) : trace(apply_blockTM_l(M, TM, 2))
        
        TM = applyTM_op_nn_l(A,A,A,A, op, TM)
        
        return trace(TM) / normsq
    end
end

"""
    expect{T}(M::puMPState{T}, op::MPO_open{T}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_TM{T}}=MPS_TM{T}[])

Computes the expectation value of an MPO. The MPO may have between 1 and `num_sites(M)` sites.
If it has the maximum number of sites, it is allowed to have open or periodic boundary conditions.
Otherwise the MPO bond dimension must go to 1 at both ends.
See MPS for the definition of `MPO_open`.

Optionally uses precomputed powers of the transfer matrix as `blkTMs`.
In case `MPS_is_normalized == false` computes the norm of `M` at the same time.
"""
function expect{T}(M::puMPState{T}, op::MPO_open{T}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_TM{T}}=MPS_TM{T}[])
    N = num_sites(M)
    A = mps_tensor(M)
    D = bond_dim(M)
    
    Nop = length(op)
    
    if N == Nop
        TMop = TM_dense_MPO(M, op)
        res = trace(TMop)

        if !MPS_is_normalized
            normsq = length(blkTMs) == N ? trace(blkTMs[N]) : norm(M)^2
            res /= normsq
        end
    else
        TM = length(blkTMs) >= N-Nop ? blkTMs[N-Nop] : blockTM_dense(M, N-Nop)
        
        TMop = applyTM_MPO_l(M, op, TM)
        
        res = trace(TMop)
        
        if !MPS_is_normalized
            normsq = length(blkTMs) == N ? trace(blkTMs[N]) : trace(apply_blockTM_l(M, TM, Nop))
            res /= normsq
        end
    end
    
    res
end

"""
    expect{T}(M::puMPState{T}, O::MPO_PBC_uniform{T})

Computes the expectation value of a global MPO with periodic boundary conditions.
See MPS for the definition of `MPO_PBC_uniform`.
"""
function expect{T}(M::puMPState{T}, O::MPO_PBC_uniform{T})
    N = num_sites(M)
    OB, OM = O    
    O_full = MPOTensor{T}[OB, (OM for j in 2:N)...]
    expect(M, O_full)
end

"""
    MPS.TM_dense_MPO{T}(M::puMPState{T}, O::MPO_open{T})::MPS_MPO_TM{T}

The transfer matrix for the entire length of the MPO `O`.
"""
function MPS.TM_dense_MPO{T}(M::puMPState{T}, O::MPO_open{T})::MPS_MPO_TM{T}
    A = mps_tensor(M)
    applyTM_MPO_l(M, O[2:end], TM_dense_MPO(A, A, O[1]))
end   

"""
    blockTMs_MPO{T}(M::puMPState{T}, O::MPO_PBC_uniform{T}, N::Int=num_sites(M))

Block transfer matrices for the MPO `O` for sites 1 to n, with `n in 1:N`.
These are not all powers of the same transfer matrix, since the MPO with PBC
is generally not completely uniform.
"""
function blockTMs_MPO{T}(M::puMPState{T}, O::MPO_PBC_uniform{T}, N::Int=num_sites(M))
    A = mps_tensor(M)
    OB, OM = O
    
    TMs = MPS_MPO_TM{T}[TM_dense_MPO(A, A, OM)]
    
    work = Vector{T}(0)
    for n in 2:N-1
        TMres = res_applyTM_MPO_l(A, A, OM, TMs[end])
        work = workvec_applyTM_MPO_l!(work, A, A, OM, TMs[end])
        push!(TMs, applyTM_MPO_l!(TMres, A, A, OM, TMs[end], work))
    end
    
    #We choose to put the boundary tensors at the end to aid flexibility
    TMres = res_applyTM_MPO_l(A, A, OB, TMs[end])
    work = workvec_applyTM_MPO_l!(work, A, A, OB, TMs[end])
    push!(TMs, applyTM_MPO_l!(TMres, A, A, OB, TMs[end], work))
    
    TMs
end

"""
    MPS.applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}; 

Apply an MPS-MPO transfer matrix `TM1` derived from the puMPState `M` and the MPO `O` to an
exisiting MPS-MPO transfer matrix `TM2`, acting to the left: `TM2 * TM1`.

A vector for holding intermediate results may be supplied as `work`. It may be resized!
"""
function MPS.applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}; 
    work::Vector{T}=Vector{T}())::MPS_MPO_TM{T}

    A = mps_tensor(M)
    
    TM = TM2
    if length(O) > 0
        TMres = res_applyTM_MPO_l(A, A, O[1], TM)
        for n in 1:length(O)
            TMres = size(O[n],1) != size(O[n], 3) ? res_applyTM_MPO_l(A, A, O[n], TM) : TMres
            workvec_applyTM_MPO_l!(work, A, A, O[n], TM)
            TM = applyTM_MPO_l!(TMres, A, A, O[n], TM, work)
        end
    end
    TM
end

"""
    MPS.res_applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T})

Prepare appropriately-sized result arrays `TMres` to hold the intermediate and final results of 
`applyTM_MPO_l!(TMres, M, O, TM2)`.
"""
function MPS.res_applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T})
    A = mps_tensor(M)
    res = MPS_MPO_TM{T}[]
    for n in 1:length(O)
        push!(res, res_applyTM_MPO_l(A, A, O[n], TM2))
        TM2 = res[end]
    end
    res
end

"""
    MPS.workvec_applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2s::Vector{MPS_MPO_TM{T}})

Prepare working-memory vector for applyTM_MPO_l!().
"""
function MPS.workvec_applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2s::Vector{MPS_MPO_TM{T}})
    A = mps_tensor(M)
    len = 0
    for j in 1:length(O)
        len = max(len, worklen_applyTM_MPO_l(A, A, O[j], TM2s[j]))
    end
    workMPO = Vector{T}(len)
end

"""
    MPS.applyTM_MPO_l!{T}(TMres::Vector{MPS_MPO_TM{T}}, M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T}

Apply an MPS-MPO transfer matrix `TM1` derived from the puMPState `M` and the MPO `O` to an
exisiting MPS-MPO transfer matrix `TM2`, acting to the left:
    `TM2 * TM1`
This version accepts a vector `work` for working memory as well as a preallocated
set of result arrays `TMres`.
"""
function MPS.applyTM_MPO_l!{T}(TMres::Vector{MPS_MPO_TM{T}}, M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T}
    A = mps_tensor(M)
    D = bond_dim(M) 
    
    TM = TM2
    for n in 1:length(O)
        applyTM_MPO_l!(TMres[n], A, A, O[n], TM, work)
        TM = TMres[n]
    end
    TM
end

"""
    MPS.applyTM_MPO_r{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}; work::Vector{T}=Vector{T}())::MPS_MPO_TM{T}

Apply an MPS-MPO transfer matrix `TM1` derived from the puMPState `M` and the MPO `O` to an
exisiting MPS-MPO transfer matrix `TM2`, acting to the right: `TM1 * TM2`.

A vector for holding intermediate results may be supplied as `work`. It may be resized!
"""
function MPS.applyTM_MPO_r{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}; 
    work::Vector{T}=Vector{T}())::MPS_MPO_TM{T}
    A = mps_tensor(M)
    
    TM = TM2
    if length(O) > 0
        TMres = res_applyTM_MPO_r(A, A, O[end], TM)
        for n in length(O):-1:1
            Mres = size(O[n],1) != size(O[n], 3) ? res_applyTM_MPO_r(A, A, O[n], TM) : TMres
            workvec_applyTM_MPO_r!(work, A, A, O[n], TM)
            TM = applyTM_MPO_r!(TMres, A, A, O[n], TM, work)
        end
    end
    TM
end

function MPS.applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_TM{T}; work::Vector{T}=Vector{T}())::MPS_TM{T}
    TM_convert(applyTM_MPO_l(M, O, TM_convert(TM2), work=work))
end

MPS.res_applyTM_MPO_l{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_TM{T}) = res_applyTM_MPO_l(M, O, TM_convert(TM2))

function MPS.applyTM_MPO_l!{T}(TMres::Vector{MPS_MPO_TM{T}}, M::puMPState{T}, O::MPO_open{T}, TM2::MPS_TM{T}, work::Vector{T})::MPS_TM{T}
    TM_convert(applyTM_MPO_l!(TMres, M, O, TM_convert(TM2), work))
end

function MPS.applyTM_MPO_r{T}(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_TM{T}; work::Vector{T}=Vector{T}())::MPS_TM{T}
    TM_convert(applyTM_MPO_r(M, O, TM_convert(TM2), work=work))
end

"""
    derivatives_1s{T}(M::puMPState{T}, h::MPO_open{T}; blkTMs::Vector{MPS_TM{T}}=blockTMs(M, num_sites(M)-1), e0::Float64=0.0)

This returns the energy derivatives with respect to the elements of the conjugate `conj(A)` of one
tensor of the MPS `M`. This the same as the result of applying the effective Hamiltonian for one 
tensor `A` of the puMPState `M` to the current value of `A`: It is `H_eff * vec(A)`.

The Hamiltonian is assumed to be a sum of local terms equal to eachother up to translation. 
The local term is supplied as an MPO `h`.

The energy density of the state `M` with respect to the Hamiltonian may be supplied as `e0`.
If supplied, it is used to subtract the contribution of the derivatives that change only the norm/phase
of the state: The Hamiltonian `H` becomes `H - e0 * I`.

Pre-computed powers of the transfer matrix may be supplied as `blkTMs` to avoid recomputing them.
"""
function derivatives_1s{T}(M::puMPState{T}, h::MPO_open{T}; blkTMs::Vector{MPS_TM{T}}=blockTMs(M, num_sites(M)-1), e0::Float64=0.0)
    A = mps_tensor(M)
    N = num_sites(M)
    D = bond_dim(M)
    
    j = 1
    TM = blkTMs[j]
    
    #Transfer matrix with one H term
    TM_H = TM_convert(TM_dense_MPO(M, h))
    
    #Subtract energy density e0 * I.
    #Note: We do this for each h term individually in order to avoid a larger subtraction later.
    #Assumption: This is similarly accurate to subtracting I*e0 from the Hamiltonian itself.
    #The transfer matrices typically have similar norm before and after subtraction, even when
    #the final gradient has small physical norm.
    LinAlg.axpy!(-e0, blkTMs[length(h)], TM_H) 
    
    TM_H_res = similar(TM_H)    
    
    work = workvec_applyTM_l(A, A)
    
    TMMPO_res = res_applyTM_MPO_l(M, h, TM)
    workMPO = workvec_applyTM_MPO_l(M, h, vcat(MPS_MPO_TM{T}[TM_convert(TM)], TMMPO_res[1:end-1]))
    
    for k in length(h)+1:N-1 #leave out one site (where we take the derivative)
        #Extend TM_H
        applyTM_l!(TM_H_res, A, A, TM_H, work)
        TM_H, TM_H_res = (TM_H_res, TM_H)
        
        #New H term
        TM_H_add = applyTM_MPO_l!(TMMPO_res, M, h, TM, workMPO)
        BLAS.axpy!(-e0, blkTMs[j+length(h)], TM_H_add) #Subtract energy density e0 * I
        
        j += 1
        TM = blkTMs[j]
        
        BLAS.axpy!(1.0, TM_H_add, TM_H) #add new H term to TM_H
    end
    
    #effective ham terms that do not act on gradient site
    LinAlg.axpy!(-length(h)*e0, blkTMs[N-1], TM_H) #Subtract energy density for the final terms

    #Add only the A, leaving a conjugate gap.
    @tensor d_A[l, s, r] := A[k1, s, k2] * TM_H[k2,r, k1,l]
    
    #NOTE: TM now has N-length(h) sites
    TM = TM_convert(TM)
    for n in 1:length(h)
        TM_H = applyTM_MPO_l(M, h[1:n-1], TM, work=workMPO)
        TM_H = applyTM_MPO_r(M, h[n+1:end], TM_H, work=workMPO)
        hn = h[n]
        @tensor d_A[l, t, r] += (A[k1, s, k2] * TM_H[k2,m2,r, k1,m1,l]) * hn[m1,s,m2,t] #allocates temporaries
    end
    
    d_A
end

function BiCGstab(M,V,X0,tol::Float64; max_itr::Int=100)
    #use BiCGSTAB, solve MX=V,with guess X0
    d = length(V)
    r = V - M*X0
    r_tilde = copy(r)
    x = copy(X0)
    rho2 = 1
    alpha = 1
    omega = 1

    v = similar(V)
    s = similar(V)
    t = similar(V)
    norm_V = norm(V)
    p = copy(r)
    
    converged = false
    for i in 1:max_itr
        rho1 = dot(r_tilde,r)
        if i > 1
            beta = rho1/rho2 * alpha/omega
            p .= r .+ beta .* (p .- omega .* v)
        end
        v = M*p
        alpha = rho1/(dot(r_tilde,v))
        s .= r .- alpha .* v
        # output condition 1
        norm_s = norm(s)
        if norm_s/norm_V < tol
            x .= x .+ alpha .* p
            converged = true
            break
        end
        t = M*s
        omega = dot(t,s)/dot(t,t)
        x .= x .+ alpha .* p .+ omega .* s
        r .= s .- omega .* t
        norm_r = norm(r)
        if norm_r/norm_V < tol
            converged = true
            break
        end
        rho2 = rho1
    end 
    
    !converged && warn("BiCGStab did not converge (tol: $tol, max_itr: $max_itr)")
    
    x
end

# try
#     @pyimport scipy.sparse.linalg as SLA
# catch
#     warn("Could not import sparse linear algebra from Scipy.")
#     SLA = nothing
# end

# function BiCGstab_scipy(M,V,X0,tol::Float64; max_itr::Int=100, max_attempts::Int=1)
#     res = nothing
#     for j in 1:max_attempts
#         res, info = SLA.bicgstab(M, V, X0, tol, maxiter=max_itr)
#         info < 0 && error("BiCGStab failed due to illegal input or breakdown")
#         if info > 0
#             warn("BiCGStab did not converge (tol: $tol, max_itr: $max_itr), attempt $j of $max_attempts")
#             j < max_attempts && rand!(X0) #NOTE: This does not seem to help much, hence max_attempts=1 by default.
#         else
#             break
#         end
#     end
#     res
# end

# function lGMRes_scipy(M,V,X0,tol::Float64; max_itr::Int=100)
#     res, info = SLA.lgmres(M, V, X0, tol, maxiter=max_itr)
#     info < 0 && error("lGMRes failed due to illegal input or breakdown")
#     info > 0 && warn("lGMRes did not converge (tol: $tol, max_itr: $max_itr)")
#     res
# end

"""
    gradient_central{T}(M::puMPState{T}, inv_lambda::AbstractMatrix{T}, d_A::MPSTensor{T};
        sparse_inverse::Bool=true, pinv_tol::Float64=1e-12, 
        max_itr::Int=500, tol::Float64=1e-12,
        grad_Ac_init::MPSTensor{T}=rand_MPSTensor(T, phys_dim(M), bond_dim(M)),
        blkTMs::Vector{MPS_TM{T}}=MPS_TM{T}[])

Converts the energy derivatives supplied by `derivatives_1s` into the energy gradient for
a single tensor of the puMPState `M`.

We first do a gauge transformation using `inv_lambda` to bring one tensor into the centre gauge.
This makes the inverse of the induced physical metric `Nc` on the one-site tensor parameters, which
is needed to compute the gradient from the derivatives, better conditioned.

If `sparse_inverse == false`, the inverse is computed explicitly as a pseudo inverse at cost O(`bond_dim(M)^6`).
Otherwise it is computed implicitly using the BiCGStab solver at cost O(`bond_dim(M)^4 * num_iter`).

The physical norm of the gradient is also computed and returned.
"""
function gradient_central{T}(M::puMPState{T}, inv_lambda::AbstractMatrix{T}, d_A::MPSTensor{T}; 
        sparse_inverse::Bool=true, pinv_tol::Float64=1e-12, 
        max_itr::Int=500, tol::Float64=1e-12,
        grad_Ac_init::MPSTensor{T}=rand_MPSTensor(T, phys_dim(M), bond_dim(M)),
        blkTMs::Vector{MPS_TM{T}}=MPS_TM{T}[])
    N = num_sites(M)
    D = bond_dim(M)
    d = phys_dim(M)
    
    inv_lambda = full(inv_lambda)
    
    T1 = length(blkTMs) >= N-1 ? blkTMs[N-1] : blockTM_dense(M, N-1)
    
    #Overlap matrix in central gauge (except for the identity on the physical dimension)
    Nc = ncon((inv_lambda, inv_lambda, T1), ((-4,1), (-2,2), (1,2,-3,-1)))
    Nc = reshape(Nc, (D^2, D^2))
    ## Note that above can also be obtained from the normalization process
    
    d_Ac = ncon((d_A, inv_lambda), ((-1,-3,1), (1,-2))) # now size (D,D,d)
    
    grad_Ac_init = permutedims(grad_Ac_init, (1,3,2)) # now size (D,D,d)
    
    grad_Ac = zeros(d_Ac)
    
    if sparse_inverse
        #Split the inverse problem along the physical dimension, since N acts trivially on that factor. Avoids constructing N x I.
        for s in 1:d
            grad_vec = BiCGstab(Nc, vec(view(d_Ac, :,:,s)), vec(view(grad_Ac_init, :,:,s)), tol, max_itr=max_itr)
            copy!(view(grad_Ac, :,:,s), grad_vec)
        end
    else
        #Dense version
        #Nc_i = inv(Nc)
        Nc_i = pinv(Nc, pinv_tol)
        for s in 1:d
            grad_vec = Nc_i * vec(view(d_Ac, :,:,s))
            copy!(view(grad_Ac, :,:,s), grad_vec)
        end
    end
    
    grad_A = ncon((grad_Ac, inv_lambda), ([-1,1,-2],[1,-3])) # back to (D,d,D)
    
    norm_grad_A = sqrt(abs(dot(vec(grad_A), vec(d_A))))
    
    grad_A, norm_grad_A, permutedims(grad_Ac, (1,3,2))
end

type EnergyHighException <: Exception
    stp::Float64
    En::Float64
end
type WolfeAbortException <: Exception 
    stp::Float64
    En::Float64
end

"""
    line_search_energy{T}(M::puMPState{T}, En0::Float64, grad::MPSTensor{T}, grad_normsq::Float64, step::Float64, hMPO::MPO_open{T}; itr::Int=10, rel_tol::Float64=1e-1, max_attempts::Int=3, wolfe_c1::Float64=100.0)

Conducts a line search starting at the puMPState `M` to find the puMPState closest to the energetic minimum along 
the search-direction specified by `grad`.

`En0` should contain the energy-density of `M`, which will be used as a refernce point: 
Steps that increase the energy are avoided, although not completely excluded. 
Where they occur, they will typically be small compared to `step`.

`step` is a guide for the initial step length.

`hMPO` is the local Hamiltonian term in MPO form. It is used to compute the energy density.
"""
function line_search_energy{T}(M::puMPState{T}, En0::Float64, grad::MPSTensor{T}, grad_normsq::Float64, step::Float64, hMPO::MPO_open{T}; itr::Int=10, rel_tol::Float64=1e-1, max_attempts::Int=3, wolfe_c1::Float64=100.0)
    M_new = copy(M)
    num_calls::Int = 0
    attempt::Int = 0
    
    f = (stp::Float64)->begin
        num_calls += 1
        
        set_mps_tensor!(M_new, mps_tensor(M) .- stp .* grad)
        
        En = real(expect(M_new, hMPO, MPS_is_normalized=false)) #computes the norm and energy-density in one step
        
        println("Linesearch: $stp, $En")

        #Abort the search if the first step already increases the energy compared to the initial state
        num_calls == 1 && En > En0 && throw(EnergyHighException(stp, En))
        
        #Note: This is the first Wolfe condition, plus a minimum step size, since we don't want to compute the gradient...
        #Probably it effectively only serves to reduce the maximum step size reached, thus we turn it off by setting wolfe_c1=100.
        stp > 1e-2 && En <= En0 - wolfe_c1 * stp * grad_normsq && throw(WolfeAbortException(stp, En))
        
        En
    end
    
    res = nothing
    while attempt <= max_attempts
        try
            attempt += 1
            ores = optimize(f, step/5, step*1.8, Brent(), iterations=itr, rel_tol=rel_tol, store_trace=false, extended_trace=false)
            res = Optim.minimizer(ores), Optim.minimum(ores)
            break
        catch e
            if isa(e, EnergyHighException)
                if attempt < max_attempts
                    warn("Linesearch: Initial step was too large. Adjusting!")
                    step *= 0.1
                    num_calls = 0
                else
                    warn("Linesearch: Initial step was too large. Aborting!")
                    res = e.stp, e.En
                    break
                end
            elseif isa(e, WolfeAbortException)
                info("Linesearch: Early stop due to good enough step!")
                res = e.stp, e.En
                break
            else
                rethrow(e)
            end
        end
    end
    
    res
end

"""
    minimize_energy_local!{T}(M::puMPState{T}, hMPO::MPO_open{T}, itr::Int; 
        step::Float64=0.001, 
        grad_max_itr::Int=500,
        grad_sparse_inverse::Bool=true)

Optimises the puMPState `M` to minimize the energy with respect to a translation-invariant local Hamiltonian.
The local Hamiltonian term is supplied as an open MPO `hMPO`, which is a vector of `MPOTensor`:
`hMPO = MPOTensor[h1,h2,...,hn]`.

This MPO has a range of `n` sites. The `MPOTensor`s `h1` and `hn` must have outer MPO bond dimension 1.
For a nearest-neighbour Hamiltonian, `n=2`.
"""
function minimize_energy_local!{T}(M::puMPState{T}, hMPO::MPO_open{T}, itr::Int; 
        step::Float64=0.001, 
        grad_max_itr::Int=500,
        grad_sparse_inverse::Bool=false)
    blkTMs = blockTMs(M)
    normalize!(M, blkTMs)
    En = real(expect(M, hMPO, blkTMs=blkTMs))
    
    grad_Ac = rand_MPSTensor(T, phys_dim(M), bond_dim(M)) #Used to initialise the BiCG solver
    stol = 1e-12
    
    for k in 1:itr
        println("Itr: $k")
        @time M, lambda, lambda_i = canonicalize_left!(M)
        
        @time blkTMs = blockTMs(M)
        @time deriv = derivatives_1s(M, hMPO, blkTMs=blkTMs, e0=En)
        @time grad, norm_grad, grad_Ac = gradient_central(M, lambda_i, deriv, sparse_inverse=grad_sparse_inverse, grad_Ac_init=grad_Ac, blkTMs=blkTMs, tol=stol, max_itr=grad_max_itr)
        
        stol = min(1e-6, max(norm_grad^2/10, 1e-12))
        En_prev = En
        @time step, En = line_search_energy(M, En, grad, norm_grad^2, min(max(step, 0.001),0.1), hMPO)
        println("$norm_grad, $step, $En, $(En-En_prev)")
        if norm_grad < 1e-6
            break
            @show k
        end
        
        Anew = mps_tensor(M) .- step .* grad
        set_mps_tensor!(M, Anew)
        @time normalize!(M)
    end
    
    normalize!(M)
    M
end

#--------------

"""
    Represents a tangent vector, with momentum `p=2π/N*k`, living in the tangent space of a puMPState `state`.
"""
type puMPSTvec{T}
    state::puMPState{T}
    B::MPSTensor{T}
    k::Int #p=2π/N*k
end

#For holding lots of Tvecs for the same state and momentum
type puMPSTvecs{T}
    state::puMPState{T}
    Bs::Array{T,4}
    k::Int #p=2π/N*k
end

Base.copy(Tvec::puMPSTvec) = puMPSTvec(Tvec.state, copy(Tvec.B), Tvec.k, Tvec.N) #does not copy state

MPS.bond_dim(Tvec::puMPSTvec) = bond_dim(Tvec.state)
MPS.phys_dim(Tvec::puMPSTvec) = phys_dim(Tvec.state)
state(Tvec::puMPSTvec) = Tvec.state
tvec_tensor(Tvec::puMPSTvec) = Tvec.B
mps_tensors(Tvec::puMPSTvec) = (mps_tensor(Tvec.state), tvec_tensor(Tvec))
num_sites(Tvec::puMPSTvec) = num_sites(Tvec.state)
spin(Tvec::puMPSTvec) = Tvec.k 
momentum(Tvec::puMPSTvec) = 2π/num_sites(Tvec) * spin(Tvec)

set_tvec_tensor!{T}(Tvec::puMPSTvec{T}, B::MPSTensor{T}) = Tvec.B = B

"""
    Base.norm{T}(Tvec::puMPSTvec{T})

Computes the norm of a puMPState tangent vector `Tvec`.
"""
function Base.norm{T}(Tvec::puMPSTvec{T})
    N = num_sites(Tvec)
    A, B = mps_tensors(Tvec)
    p = momentum(Tvec)
    
    #conj(B) at fixed location
    TBs = TM_dense(B, B)
    TA = TM_dense(A, B)
    
    TMres = similar(TA)
    work = workvec_applyTM_l(A, A, TA)
    for n in 2:N
        TBs = applyTM_l!(TBs, A, A, TBs, work)
        
        TAB = applyTM_l!(TMres, B, A, TA, work)
        BLAS.axpy!(cis(p*n), TAB, TBs) #Add to complete terms (containing both B's)
        
        TA = applyTM_l!(TA, A, A, TA, work)
    end
    
    sqrt(N*trace(TBs))
end

"""
    Base.normalize!{T}(Tvec::puMPSTvec{T}) = scale!(Tvec.B, 1.0/norm(Tvec))

Normalizes a puMPState tangent vector `Tvec` in place.
"""
Base.normalize!{T}(Tvec::puMPSTvec{T}) = scale!(Tvec.B, 1.0/norm(Tvec))

"""
    expect{T}(Tvec::puMPSTvec{T}, O::MPO_PBC_uniform{T})

The expectation value of a translation-invariant operator specified as an MPO
with periodic boundary conditions, with respect to the puMPState tangent vector `Tvec`.
"""
function expect{T}(Tvec::puMPSTvec{T}, O::MPO_PBC_uniform{T})
    A, B = mps_tensors(Tvec)
    OB, OM = O
    
    N = num_sites(Tvec)
    p = momentum(Tvec)

    #conj(B) at fixed location...
    TBs = TM_dense_MPO(B, B, OB)
    TA = TM_dense_MPO(A, B, OB)
    
    TMres = similar(TA)
    work = workvec_applyTM_MPO_l(A, A, OM, TA)
    for n in 2:N
        TBs = applyTM_MPO_l!(TBs, A, A, OM, TBs, work)
        
        TAB = applyTM_MPO_l!(TMres, B, A, OM, TA, work)
        BLAS.axpy!(cis(p*(n-1)), TAB, TBs) #Add to complete terms (containing both B's)
        
        n < N && (TA = applyTM_MPO_l!(TA, A, A, OM, TA, work))
    end
    
    N*trace(TBs)
end

#Slower version of the above
function expect_using_overlap{T}(Tvec::puMPSTvec{T}, O::MPO_PBC_uniform{T})
    OB, OM = O
    expect(Tvec, MPOTensor{T}[OB, (OM for j in 1:num_sites(Tvec)-1)...])
end

"""
    expect_global_product{T,TO}(Tvec::puMPSTvec{T}, O::Matrix{TO})

The expectation value of a global product of on-site operators `O` with
respect to the puMPState tangent vector `Tvec`.
"""
function expect_global_product{T,TO}(Tvec::puMPSTvec{T}, O::Matrix{TO})
    Or = reshape(O.', (1,size(O,2),1,size(O,1)))
    O_MPO = (Or,Or)
    expect(Tvec, O_MPO)
end

"""
    expect{T}(Tvec::puMPSTvec{T}, O::MPO_open{T})

The expectation value of an arbitrary MPO with respect to the puMPState tangent
vector `Tvec`.

Note: This is not the most efficient implementation, since it uses `overlap()` and 
      does not take advantage of the two tangent vectors being the same.
"""
function expect{T}(Tvec::puMPSTvec{T}, O::MPO_open{T})
    overlap(Tvec, O, Tvec)
end

"""
    overlap{T}(Tvec2::puMPSTvec{T}, O::MPO_open{T}, Tvec1::puMPSTvec{T})

Computes the physical-space inner product `<Tvec2 | O | Tvec1>`,
where `Tvec1` and `Tvec2` are puMPState tangent vectors and `O` is an operator in MPO form.
The tangent vectors need not live in the same puMPState tangent space, since the inner product
is taken in the physical space. They must have the same number of physical sites.

The operator MPO may have between 1 and `N = num_sites()` tensors. 
In case it has `N` tensors, it may have a bond around the circle, between sites `N` and 1.
"""
function overlap{T}(Tvec2::puMPSTvec{T}, O::MPO_open{T}, Tvec1::puMPSTvec{T}; 
    TAA_all::Vector{Array}=Array[], TBAs_all::Vector{Array}=Array[])

    A1, B1 = mps_tensors(Tvec1)
    A2, B2 = mps_tensors(Tvec2)
    
    N = num_sites(Tvec1)
    @assert num_sites(Tvec2) == N
    
    p1 = momentum(Tvec1)
    p2 = momentum(Tvec2)
    
    #Fix site numbering with O at sites 1..length(O)
    
    TAA = length(TAA_all) > 0 ? TAA_all[1] : TM_dense_MPO(A1, A2, O[1])
    
    TBBs = TM_dense_MPO(B1, B2, O[1])
    scale!(TBBs, cis(p1-p2))
    
    TABs = TM_dense_MPO(A1, B2, O[1])
    scale!(TABs, cis(-p2))
    
    if length(TBAs_all) > 0
        TBAs = TBAs_all[1]
    else
        TBAs = TM_dense_MPO(B1, A2, O[1])
        scale!(TBAs, cis(p1))
    end
    
    work = Vector{T}()
    for n in 2:length(O)
        work = workvec_applyTM_MPO_l!(work, A1, A2, O[n], TAA)
        TMres = res_applyTM_MPO_l(A1, A2, O[n], TAA)
        
        TBBs = applyTM_MPO_l!(similar(TMres), A1, A2, O[n], TBBs, work)
        
        BLAS.axpy!(cis(n*(p1-p2)), applyTM_MPO_l!(TMres, B1, B2, O[n], TAA, work), TBBs)
        BLAS.axpy!(cis(n*(-p2)), applyTM_MPO_l!(TMres, A1, B2, O[n], TBAs, work), TBBs)
        BLAS.axpy!(cis(n*(p1)), applyTM_MPO_l!(TMres, B1, A2, O[n], TABs, work), TBBs)
        
        if n < N
            TABs = applyTM_MPO_l!(similar(TMres), A1, A2, O[n], TABs, work)
            BLAS.axpy!(cis(n*(-p2)), applyTM_MPO_l!(TMres, A1, B2, O[n], TAA, work), TABs)

            if length(TBAs_all) > 0
                TBAs = TBAs_all[n]
            else
                TBAs = applyTM_MPO_l!(similar(TMres), A1, A2, O[n], TBAs, work)
                BLAS.axpy!(cis(n*(p1)), applyTM_MPO_l!(TMres, B1, A2, O[n], TAA, work), TBAs)
            end

            TAA = length(TAA_all) > 0 ? TAA_all[n] : applyTM_MPO_l!(similar(TMres), A1, A2, O[n], TAA, work)
        end
    end
    
    if length(O) < N
        TBBs = TM_convert(TBBs)
        TABs = TM_convert(TABs)
        TBAs = TM_convert(TBAs)
        TAA = TM_convert(TAA)
        
        TMres = similar(TAA)
        for n in length(O)+1:N
            work = workvec_applyTM_l!(work, A1, A2, TAA)
            
            TBBs = applyTM_l!(TBBs, A1, A2, TBBs, work)

            BLAS.axpy!(cis(n*(p1-p2)), applyTM_l!(TMres, B1, B2, TAA, work), TBBs)
            BLAS.axpy!(cis(n*(-p2)), applyTM_l!(TMres, A1, B2, TBAs, work), TBBs)
            BLAS.axpy!(cis(n*(p1)), applyTM_l!(TMres, B1, A2, TABs, work), TBBs)

            if n < N
                TABs = applyTM_l!(TABs, A1, A2, TABs, work)
                BLAS.axpy!(cis(n*(-p2)), applyTM_l!(TMres, A1, B2, TAA, work), TABs)
                
                if length(TBAs_all) > 0
                    TBAs = TBAs_all[n]
                else
                    TBAs = applyTM_l!(TBAs, A1, A2, TBAs, work)
                    BLAS.axpy!(cis(n*(p1)), applyTM_l!(TMres, B1, A2, TAA, work), TBAs)
                end
                
                TAA = length(TAA_all) > 0 ? TAA_all[n] : applyTM_l!(TAA, A1, A2, TAA, work)
            end
        end
    end
    
    trace(TBBs)
end

"""
    overlap_precomp_samestate{T}(O::MPO_open{T}, Tvec1::puMPSTvec{T})

Precompute some data for `overlap()`, useful for cases where many overlaps are computed
with tangent vectors living in the same tangent space.
"""
function overlap_precomp_samestate{T}(O::MPO_open{T}, Tvec1::puMPSTvec{T})
    A1, B1 = mps_tensors(Tvec1)
    p1 = momentum(Tvec1)
    N = num_sites(Tvec1)

    TAA_all = Array[]
    TBAs_all = Array[]

    TAA = TM_dense_MPO(A1, A1, O[1])
    push!(TAA_all, TAA)
    TBAs = TM_dense_MPO(B1, A1, O[1])
    scale!(TBAs, cis(p1))
    push!(TBAs_all, TBAs)

    work = Vector{T}()
    for n in 2:min(length(O), N-1)
        work = workvec_applyTM_MPO_l!(work, A1, A1, O[n], TAA)
        TMres = res_applyTM_MPO_l(A1, A1, O[n], TAA)

        TBAs = applyTM_MPO_l!(similar(TMres), A1, A1, O[n], TBAs, work)
        BLAS.axpy!(cis(n*(p1)), applyTM_MPO_l!(TMres, B1, A1, O[n], TAA, work), TBAs)
        push!(TBAs_all, TBAs)

        TAA = applyTM_MPO_l!(similar(TMres), A1, A1, O[n], TAA, work)
        push!(TAA_all, TAA)
    end

    if length(O) < N
        TBAs = TM_convert(TBAs)
        TAA = TM_convert(TAA)
        
        TMres = similar(TAA)
        for n in length(O)+1:N-1
            work = workvec_applyTM_l!(work, A1, A1, TAA)

            TBAs = applyTM_l!(TBAs, A1, A1, TBAs, work)
            BLAS.axpy!(cis(n*(p1)), applyTM_l!(TMres, B1, A1, TAA, work), TBAs)
            push!(TBAs_all, TBAs)
            
            TAA = applyTM_l!(TAA, A1, A1, TAA, work)
            push!(TAA_all, TAA)
        end
    end

    TAA_all, TBAs_all
end

"""
    op_in_basis{T}(Tvec_basis::Vector{puMPSTvec{T}}, O::MPO_open{T})

Compute the matrix elements of the operator `O` in the basis of tangent vectors `Tvec_basis`.
"""
function op_in_basis{T}(Tvec_basis::Vector{puMPSTvec{T}}, O::MPO_open{T})
    Nvec = length(Tvec_basis)
    
    M = state(Tvec_basis[1])
    @assert all(state(tv) === M for tv in Tvec_basis)

    O_in_basis = zeros(T, (Nvec, Nvec))
    for j in 1:Nvec
        #Increases memory usage by approx. 2*N*D^4
        TAA_all, TBAs_all = overlap_precomp_samestate(O, Tvec_basis[j])
        for k in 1:Nvec
            O_in_basis[k,j] = overlap(Tvec_basis[k], O, Tvec_basis[j]; TAA_all=TAA_all, TBAs_all=TBAs_all)
        end
    end
    O_in_basis
end

"""
    tangent_space_metric{T}(M::puMPState{T}, ks::Vector{Int}, lambda_i::Matrix{T}, blkTMs::Vector{MPS_TM{T}})

Computes the physical metric induced on the tangent space of the puMPState `M` for the tangent-space
momentum sectors specified in `ks`. The momenta are `ps = 2π/N .* ks` where `N = num_sites(M)`. 
The tangent space is parameterised such that the `B` tensors are in the centre gauge.

This assumes that the matrix `lambda_i` is the inverse of `lambda`, the gauge-transformation 
needed to convert an mps tensor `A` of `M` to the centre gauge: `Ac = A * lambda`.
"""
function tangent_space_metric{T}(M::puMPState{T}, ks::Vector{Int}, lambda_i::Matrix{T}, blkTMs::Vector{MPS_TM{T}})
    #Time cost O(N*d^2*D^6).. could save a factor of dD by implementing the sparse mat-vec operation, but then we'd add a factor Nitr
    #space cost O(N)
    A = mps_tensor(M)
    N = num_sites(M)
    d = phys_dim(M)
    ps = 2π/N .* ks
    
    Gs = Array{T,6}[ ncon((eye(d), blkTMs[N-1]),([-5,-2],[-6,-3,-4,-1])) ] #Add I on the physical index. This is the same-site term.
    for j in 2:length(ps)
        push!(Gs, copy(Gs[1]))
    end
    
    Gpart = ncon((A,conj(A), blkTMs[N-2]), ([2,-2,-4],[-3,-5,1],[-6,1,2,-1]))
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]), Gpart, Gs[j])
    end
    
    for i in 2:N-2
        left_T = ncon((A, blkTMs[i-1]), ([-1,-2,1],[1,-3,-4,-5])) #gap on the top-left, then usual TM
        right_T = ncon((conj(A), blkTMs[N-i-1]), ([-3,-2,1],[-1,1,-4,-5])) #gap on the bottom-right
        
        Gpart = ncon((left_T, right_T), ([2,-2,-3,-4,1],[-6,-5,1,2,-1])) #complete loop, cost O(d^2 * D^6)
        for j in 1:length(ps)
            BLAS.axpy!(cis(ps[j]*i), Gpart, Gs[j])
        end
    end
    
    Gpart = ncon((A, blkTMs[N-2], conj(A)), ([-6,-2,2],[2,-3,-4,1],[1,-5,-1]))
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]*(N-1)), Gpart, Gs[j])
        
        Gs[j] = ncon((Gs[j], lambda_i, lambda_i), ([-1,-2,1,-4,-5,2],[1,-3],[2,-6]))
    
        scale!(Gs[j], N)
    end
    
    Matrix{T}[reshape(G, (length(A), length(A))) for G in Gs]
end

"""
    tangent_space_metric_and_hamiltonian{T}(M::puMPState{T}, H::MPO_PBC_uniform{T}, ks::Vector{Int}, lambda_i::Matrix{T})

Computes the tangent-space metric and effective Hamiltonian given the physical-space Hamiltonian as an MPO with PBC,
for the momentum sectors specified in `ks`. The momenta are `ps = 2π/N .* ks` where `N = num_sites(M)`. 
The tangent space is parameterised such that the `B` tensors are in the centre gauge.

This assumes that the matrix `lambda_i` is the inverse of `lambda`, the gauge-transformation 
needed to convert an mps tensor `A` of `M` to the centre gauge: `Ac = A * lambda`.
"""
function tangent_space_metric_and_hamiltonian{T}(M::puMPState{T}, H::MPO_PBC_uniform{T}, ks::Vector{Int}, lambda_i::Matrix{T})
    @time Gs = tangent_space_metric(M, ks, lambda_i, blockTMs(M))
    @time Heffs = tangent_space_hamiltonian(M, H, ks, lambda_i)
    
    Gs, Heffs
end

function tangent_space_hamiltonian{T}(M::puMPState{T}, H::MPO_PBC_uniform{T}, ks::Vector{Int}, lambda_i::Matrix{T})
    #Let's do this like in tangent_space_metric(), but with MPO transfer matrices.
    A = mps_tensor(M)
    D = bond_dim(M)
    d = phys_dim(M)
    N = num_sites(M)
    ps = 2π/N .* ks
    
    blkTMs_H = blockTMs_MPO(M, H) #requires N * D^4 * M^2 bytes, with M the MPO bond dimension
    
    HB, HM = H
    #NOTE: blkTMs_H have HM right up until the full TM.. blkTMs[N] ends with HB. Each term below must add an HB somewhere.
    
    Heffs = Array{T,6}[zeros(T, (D,d,D, D,d,D)) for j in 1:length(ks)]
    
    Heff_part = ncon((HB, blkTMs_H[N-1]),([1,-5,2,-2],[-6,2,-3,-4,1,-1])) #This is the same-site term.
    for j in 1:length(ps)
        BLAS.axpy!(1.0, Heff_part, Heffs[j])
    end
    
    #This is one of the terms in which the gaps are nearest-neighbours
    blk = blkTMs_H[N-2]
    @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = HB[m3,Pt,m1,pb] * (conj(A[V2b,pb,b]) * 
                                                ((blk[V2t,m1,b, t,m2,V1b] * A[t,pt,V1t]) * HM[m2,pt,m3,Pb]))
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]), Heff_part, Heffs[j])
    end
        
    for i in 2:N-2 #i is the number of sites between conj(gap) and gap.
        #Block TM times A and an MPO tensor (we choose to put the boundary tensor HB here)
        blk = blkTMs_H[i-1]
        @tensor left_T[tl,ml,pl,bl, tr,mr,br] := HB[ml,s,mi,pl] * (A[tl,s,ti] * blk[ti,mi,bl, tr,mr,br])
        
        #Block TM times conj(A) and an MPO tensor
        blk = blkTMs_H[N-i-1]
        @tensor right_T[tl,pl,ml,bl, tr,mr,br] := HM[ml,pl,mi,t] * (conj(A[bl,t,bi]) * blk[tl,mi,bi, tr,mr,br])
        
        #Combine to form an Heff contribution at cost O(D^6)
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = (left_T[t,m1,Pb,V2b, V1t,m2,b] * right_T[V2t,Pt,m2,b, t,m1,V1b])
        for j in 1:length(ps)
            BLAS.axpy!(cis(ps[j]*i), Heff_part, Heffs[j])
        end
    end
    
    #This is the other term in which the gaps are nearest-neighbours
    blk = blkTMs_H[N-2]
    @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = HB[m3,pt,m1,Pb] * (A[V2t,pt,t] * 
                                                ((blk[t,m1,V2b, V1t,m2,b] * conj(A[b,pb,V1b])) * HM[m2,Pt,m3,pb]))
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]*(N-1)), Heff_part, Heffs[j])
        Heffs[j] = ncon((Heffs[j],lambda_i,lambda_i), ([-1,-2,1,-4,-5,2],[1,-3],[2,-6]))
        scale!(Heffs[j], N)
    end
    
    Matrix{T}[reshape(Heff, (length(A), length(A))) for Heff in Heffs]
end

"""
    tangent_space_metric_and_hamiltonian{T}(M::puMPState{T}, H::MPO_PBC_split{T}, ks::Vector{Int}, lambda_i::Matrix{T})

Computes the tangent-space metric and effective Hamiltonian given the physical-space Hamiltonian as a combination
of an open MPO, representing the Hamiltonian with open boundary conditions (OBC), and a boundary MPO.
The sum of the OBC and boundary parts of the Hamiltonian must be translation invariant.
Metrics and effective Hamiltonians are computed for the momentum sectors specified in `ks`. 
The momenta are `ps = 2π/N .* ks` where `N = num_sites(M)`. 
The tangent space is parameterised such that the `B` tensors are in the centre gauge.

This assumes that the matrix `lambda_i` is the inverse of `lambda`, the gauge-transformation 
needed to convert an mps tensor `A` of `M` to the centre gauge: `Ac = A * lambda`.
"""
function tangent_space_metric_and_hamiltonian{T}(M::puMPState{T}, H::MPO_PBC_split{T}, ks::Vector{Int}, lambda_i::Matrix{T})
    A = mps_tensor(M)
    D = bond_dim(M)
    d = phys_dim(M)
    N = num_sites(M)
    
    bTMs = blockTMs(M) #requires N * D^4 bytes
    @time Gs = tangent_space_metric(M, ks, lambda_i, bTMs)
    
    H_OBC, h_b = H
    
    Heffs = Array{T,6}[zeros(T, (D,d,D, D,d,D)) for j in 1:length(ks)]
    @time tangent_space_hamiltonian_boundary!(Heffs, M, h_b, ks, bTMs)
    blkTMs = nothing #free up this space
    @time tangent_space_hamiltonian_OBC!(Heffs, M, H_OBC, ks)
    
    for j in 1:length(ks)
        Heff = Heffs[j]
        @tensor Heff[V1b,Pb,V2b, V1t,Pt,V2t] = lambda_i[vb,V2b] * (Heff[V1b,Pb,vb, V1t,Pt,vt] * lambda_i[vt,V2t])
        scale!(Heffs[j], N)
    end
    
    Heffs = Matrix{T}[reshape(Heff, (length(A), length(A))) for Heff in Heffs]
    
    Gs, Heffs
end

"""
    tangent_space_hamiltonian_OBC!{T}(Heffs::Vector{Array{T,6}}, M::puMPState{T}, H::MPO_open_uniform{T}, ks::Vector{Int})

Compute the OBC part of the effective Hamiltonian in the momentum sectors specified in `ks`. 
When the boundary term is also included, the Hamiltonian must be translation invariant.
See `tangent_space_metric_and_hamiltonian()`.
"""
function tangent_space_hamiltonian_OBC!{T}(Heffs::Vector{Array{T,6}}, M::puMPState{T}, H::MPO_open_uniform{T}, ks::Vector{Int})
    N = num_sites(M)
    HL, HM, HR = H

    Hfull = MPOTensor{T}[HL, (HM for j in 1:N-2)..., HR]

    tangent_space_hamiltonian_OBC!(Heffs, M, Hfull, ks)
end

function tangent_space_hamiltonian_OBC!{T}(Heffs::Vector{Array{T,6}}, M::puMPState{T}, H::MPO_open{T}, ks::Vector{Int})
    A = mps_tensor(M)
    N = num_sites(M)
    ps = 2π/N .* ks

    H1s = squeeze(H[1],1) #[Pt,M,Pb]
    HNs = squeeze(H[N],3) #[M,Pt,Pb]
    
    #First construct block transfer matrices. A conjugate gap at site 1, where the Hamiltonian begins.
    #This is essentially an MPO TM, but with a physical index on the left instead of an MPO index.
    Hn = H[2]
    @tensor blk_l[V1t,P,V1b, V2t,M,V2b] := ((((A[V1t,p1,vt] * H1s[p1,m,P]) 
                                             * A[vt,p2t,V2t]) * Hn[m,p2t,M,p2b]) * conj(A[V1b,p2b,V2b]))

    #We also need block TM's for the latter part of the Hamiltonian. We will add the gap on the left later.
    #We precompute these right blocks, since we will use the intermediates
    blks_r = MPS_MPO_TM{T}[TM_dense_MPO(A, A, H[N])] #site N
    work = Vector{T}()
    for n in N-1:-1:2
        work = workvec_applyTM_MPO_r!(work, A, A, H[n], blks_r[end])
        res = res_applyTM_MPO_r(A, A, H[n], blks_r[end])
        push!(blks_r, applyTM_MPO_r!(res, A, A, H[n], blks_r[end], work))
    end
    
    #Same-site term
    blk = pop!(blks_r) #block for N-1 sites
    Hn = H[1]
    @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] := Hn[m2,Pt,m1,Pb] * blk[V2t,m1,V2b, V1t,m2,V1b]
    for j in 1:length(ks)
        BLAS.axpy!(1.0, Heff_part, Heffs[j])
    end
    
    #Nearest-neighbour term with conjugate gap on the left, gap on the right. Conjugate gap is site 1.
    blk = pop!(blks_r) #block for N-2 sites
    H1 = H[1]; Hn = H[2]
    @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = (H1[m3,pt,m1,Pb] * (A[vt,pt,V1t] * 
                                                (Hn[m1,Pt,m2,pb] * (conj(A[V2b,pb,vb]) * blk[V2t,m2,vb, vt,m3,V1b]))))
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]), Heff_part, Heffs[j])
    end
    
    #Terms separated by one or more sites
    for n in 3:N-1 #if conj(gap) is at site 1, gap is at site n
        blk_r = squeeze(pop!(blks_r), 5) #N-n sites
        Hn = H[n]
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = blk_l[vt,Pb,V2b, V1t,m1,vb1] * 
                                                  (Hn[m1,Pt,m2,pb] * (conj(A[vb1,pb,vb2]) * blk_r[V2t,m2,vb2, vt,V1b]))
        for j in 1:length(ks)
            BLAS.axpy!(cis(ps[j]*(n-1)), Heff_part, Heffs[j])
        end
        
        work = workvec_applyTM_MPO_l!(work, A, A, H[n], blk_l)
        applyTM_MPO_l!(blk_l, A, A, H[n], blk_l, work) #extend blk_l by multiplying from the right by an MPO TM
    end
    
    #Nearest-neighbour term with gap on the left, conjugate gap on the right. Conjugate gap is site 1.
    @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = (blk_l[V2t,Pb,V2b, V1t,m,vb] * conj(A[vb,p,V1b])) * HNs[m,Pt,p]
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]*(N-1)), Heff_part, Heffs[j])
    end
end

#This will return the transfer matrix (TM) from site nL to site nR, given a boundary Hamiltonian centred at the
#boundary between sites N and 1.
function blockTM_hamiltonian_boundary{T}(M::puMPState{T}, h_b::MPO_open{T}, blkTMs::Vector{MPS_TM{T}}, nL::Int, nR::Int)
    A = mps_tensor(M)
    N = num_sites(M)
    Nh = length(h_b)
    
    @assert N > Nh
    
    #Split h_b into left (starting at site 1) and right (ending at site N) parts
    h_b_L = h_b[Nh÷2+1:end]
    h_b_R = h_b[1:Nh÷2]
    
    NhL = length(h_b_L)
    NhR = length(h_b_R)
    
    nL_inh = nL - (N-NhR)
    nR_inh = nR - (N-NhR)
    
    Nmid = min(nR, N-NhR) - max(nL, NhL + 1) + 1 #length of middle TM (no h_b terms)
    
    if Nmid > 0
        blk = TM_convert(blkTMs[Nmid])

        #these will add Hamiltonian terms if needed
        blk = applyTM_MPO_r(M, h_b_L[nL:end], blk)
        blk = applyTM_MPO_l(M, h_b_R[1:nR_inh], blk)
    elseif nR <= length(h_b_L)
        blk = TM_dense_MPO(A, A, h_b_L[nR])
        blk = applyTM_MPO_r(M, h_b_L[nL:nR-1], blk)
    else
        blk = TM_dense_MPO(A, A, h_b_R[nL_inh])
        blk = applyTM_MPO_l(M, h_b_R[nL_inh+1:nR_inh], blk)
    end
    
    #@show nL, nR, Nmid, nR-(N-NhR)
    
    blk
end

#This will return the boundary Hamiltonian MPO tensor for site n. Where the Hamiltonian does not act on site n,
#return a zero-length result.
function h_term_hamiltonian_boundary{T}(M::puMPState{T}, h_b::MPO_open{T}, n::Int)
    A = mps_tensor(M)
    N = num_sites(M)
    Nh = length(h_b)
    
    @assert N > Nh
    
    h_b_L = h_b[Nh÷2+1:end]
    h_b_R = h_b[1:Nh÷2]
    
    NhL = length(h_b_L)
    NhR = length(h_b_R)
    
    if n <= length(h_b_L)
        return h_b_L[n]
    elseif n >= N-NhR+1
        return h_b_R[n-(N-NhR)]
    else
        return MPOTensor{T}((0,0,0,0)) #length zero MPO tensor
    end
end

"""
    tangent_space_hamiltonian_boundary!{T}(Heffs::Vector{Array{T,6}}, M::puMPState{T}, h_b::MPO_open{T}, ks::Vector{Int}, blkTMs::Vector{MPS_TM{T}})

Compute the boundary part of the effective Hamiltonian in the momentum sectors specified in `ks`. 
See `tangent_space_metric_and_hamiltonian()`.

This is very similar to the OBC Hamiltonian case, except that we now have a 
short MPO `h_b` centred on the boundary between sites N and 1.
"""
function tangent_space_hamiltonian_boundary!{T}(Heffs::Vector{Array{T,6}}, M::puMPState{T}, h_b::MPO_open{T}, ks::Vector{Int}, blkTMs::Vector{MPS_TM{T}})
    A = mps_tensor(M)
    N = num_sites(M)
    ps = 2π/N .* ks
    
    Nh = length(h_b)
    @assert iseven(Nh)
    
    #Check this is really OBC
    @assert size(h_b[1], 1) == 1
    @assert size(h_b[end], 3) == 1
    
    #On-site term
    blk = blockTM_hamiltonian_boundary(M, h_b, blkTMs, 2, N)
    h_b_1 = h_term_hamiltonian_boundary(M, h_b, 1)
    @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] := blk[V2t,m2,V2b, V1t,m1,V1b] * h_b_1[m1,Pt,m2,Pb]
    for j in 1:length(ks)
        BLAS.axpy!(1.0, Heff_part, Heffs[j])
    end
    
    #NN-term 1, with conjugate gap (site 1) then gap
    blk = blockTM_hamiltonian_boundary(M, h_b, blkTMs, 3, N)
    h_b_n = h_term_hamiltonian_boundary(M, h_b, 2)
    if length(h_b_n) == 0
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = h_b_1[m2,pt,m1,Pb] * (A[vt,pt,V1t] * (conj(A[V2b,Pt,vb]) 
                * blk[V2t,m1,vb, vt,m2,V1b]))
    else
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = h_b_1[m3,pt,m1,Pb] * 
        (A[vt,pt,V1t] * (h_b_n[m1,Pt,m2,pb] * (conj(A[V2b,pb,vb]) * blk[V2t,m2,vb, vt,m3,V1b])))
    end
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]), Heff_part, Heffs[j])
    end
    
    #gap at site n
    for n in 3:N-1
        blkL = blockTM_hamiltonian_boundary(M, h_b, blkTMs, 2, n-1)
        @tensor blkL_cgap[V1t,M1,P,V1b, V2t,M2,V2b] := h_b_1[M1,p,m,P] * (A[V1t,p,vt] * blkL[vt,m,V1b, V2t,M2,V2b])
        
        blkR = blockTM_hamiltonian_boundary(M, h_b, blkTMs, n+1, N)
        h_b_n = h_term_hamiltonian_boundary(M, h_b, n)
        if length(h_b_n) == 0
            @tensor blkR_gap[V1t,P,M1,V1b, V2t,M2,V2b] := conj(A[V1b,P,vb]) * blkR[V1t,M1,vb, V2t,M2,V2b]
        else
            #Note: This is never called for a nearest-neighbour boundary Hamiltonian
            @tensor blkR_gap[V1t,P,M1,V1b, V2t,M2,V2b] := h_b_n[M1,P,m,p] * (conj(A[V1b,p,vb]) * blkR[V1t,m,vb, V2t,M2,V2b])
        end
        
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = blkL_cgap[vt,m2,Pb,V2b, V1t,m1,vb] * blkR_gap[V2t,Pt,m1,vb, vt,m2,V1b]
        for j in 1:length(ks)
            BLAS.axpy!(cis(ps[j]*(n-1)), Heff_part, Heffs[j])
        end
    end
    
    #NN-term 2, with gap (site N) then conjugate gap (site 1)
    blk = blockTM_hamiltonian_boundary(M, h_b, blkTMs, 2, N-1)
    h_b_n = h_term_hamiltonian_boundary(M, h_b, N)
    if length(h_b_n) == 0
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = ((h_b_1[m2,pt,m1,Pb] * (A[V2t,pt,vt] * blk[vt,m1,V2b, V1t,m2,vb])) *
            conj(A[vb,Pt,V1b]))
    else
        @tensor Heff_part[V1b,Pb,V2b, V1t,Pt,V2t] = ((h_b_1[m3,pt,m1,Pb] * (A[V2t,pt,vt] * blk[vt,m1,V2b, V1t,m2,vb])) *
            conj(A[vb,pb,V1b])) * h_b_n[m2,Pt,m3,pb]
    end
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]*(N-1)), Heff_part, Heffs[j])
    end
end

"""
    excitations!{T}(M::puMPState{T}, H::Union{MPO_PBC_uniform{T}, MPO_PBC_split{T}}, ks::Vector{Int}, num_states::Vector{Int}; pinv_tol::Real=1e-10)

Computes eigenstates of the effective Hamiltonian obtained by projecting `H` onto the tangent space of the puMPState `M`.
This is done in the momentum sectors specified in `ks`, where each entry of `k = ks[j]` specified a momentum `k*2pi/N`,
where `N` is the number of sites. 

The number of eigenstates to be computed for each momentum sector is specified in `num_states`.

The function returns a list of energies, a list of momenta (entries of `ks`), and a list of normalized tangent vectors.
"""
function excitations!{T}(M::puMPState{T}, H::Union{MPO_PBC_uniform{T}, MPO_PBC_split{T}}, ks::Vector{Int}, num_states::Vector{Int}; pinv_tol::Real=1e-10)
    M, lambda, lambda_i = canonicalize_left!(M)
    lambda_i = full(lambda_i)
    
    @time Gs, Heffs = tangent_space_metric_and_hamiltonian(M, H, ks, lambda_i)

    excitations(M, Gs, Heffs, lambda_i, ks, num_states, pinv_tol=pinv_tol)
end

function excitations{T}(M::puMPState{T}, Gs, Heffs, lambda_i, ks::Vector{Int}, num_states::Vector{Int}; pinv_tol::Real=1e-12)
    D = bond_dim(M)
    d = phys_dim(M)
    Bshp = mps_tensor_shape(d, D)
    
    ens = Vector{T}[]
    exs = Vector{puMPSTvec{T}}[]
    ks_rep = Vector{Int}[]
    for j in 1:length(ks)
        println("k=$(ks[j])")
        G = Gs[j]
        Heff = Heffs[j]
        
        #Force Hermiticity
        G = (G + G') / 2 
        Heff = (Heff + Heff') / 2
        
        @time GiH = pinv(G, pinv_tol) * Heff
        
        @time ev, eV = eigs(GiH, nev=num_states[j])
        #@time ev, eV = eigs(Heff, G, nev=num_states[j]) #this does not deal with small eigenvalues well
        
        exs_k = puMPSTvec{T}[]
        @time for i in 1:size(eV,2)
            v = view(eV, :,i)
            Bnrm = sqrt(dot(v, G * v)) #We can use G to compute the norm
            scale!(v, 1.0/Bnrm)
            Bc_mat = reshape(v, (D*d, D))
            Bl = Bc_mat * lambda_i
            Bl = reshape(Bl, Bshp)
            push!(exs_k, puMPSTvec{T}(M, Bl, ks[j]))
        end
        
        push!(ens, ev)
        push!(exs, exs_k)
        push!(ks_rep, repeat(ks[j:j], inner=length(ev)))
    end
    
    vcat(ens...), vcat(ks_rep...), vcat(exs...)
end

"""
    tangent_space_Hn{T}(M::puMPState{T}, Hn_split::Tuple{Int, MPO_open{T}, MPO_open{T}}, ks::Vector{Int})

Given an MPO representation of a Hamiltonian Fourier mode, split into a large OBC MPO and a boundary
MPO, `Hn_split`, computes its representation on the tangent space of the puMPState `M` for the tangent
vector momenta (of the ket tangent vector) specified in `ks`.
"""
function tangent_space_Hn{T}(M::puMPState{T}, Hn_split::Tuple{Int, MPO_open{T}, MPO_open{T}}, ks::Vector{Int})
    A = mps_tensor(M)
    D = bond_dim(M)
    d = phys_dim(M)
    N = num_sites(M)

    n, Hn_OBC, Hn_b = Hn_split #Hn is a Fourier mode of the Hamiltonian density with "momentum" n * 2pi/N

    #Each entry in ks, times 2pi/N is the momentum of one of the excitations (the ket). The momentum of the
    #other excitation is automatically (ks[j] + n) * 2pi/N.
    Hn_effs = Array{T,6}[zeros(T, (D,d,D, D,d,D)) for j in 1:length(ks)]
    @time tangent_space_hamiltonian_boundary!(Hn_effs, M, Hn_b, ks, blockTMs(M))
    @time tangent_space_hamiltonian_OBC!(Hn_effs, M, Hn_OBC, ks)

    for j in 1:length(ks)
        scale!(Hn_effs[j], N)
    end
    
    Hn_effs = Matrix{T}[reshape(Hn_eff, (length(A), length(A))) for Hn_eff in Hn_effs]

    Hn_effs
end

"""
    Hn_in_basis{T}(M::puMPState{T}, Hn_split::Tuple{Int, MPO_open{T}, MPO_open{T}}, Tvec_basis::Vector{puMPSTvec{T}}, ks::Vector{Int})

Given an MPO representation of a Hamiltonian Fourier mode, split into a large OBC MPO and a boundary
MPO, `Hn_split`, computes its matrix elements in the basis of puMPState tangent vectors `Tvec_basis`
which are assumed to live in the tangent space of the puMPState `M`.
"""
function Hn_in_basis{T}(M::puMPState{T}, Hn_split::Tuple{Int, MPO_open{T}, MPO_open{T}}, Tvec_basis::Vector{puMPSTvec{T}}, ks::Vector{Int})
    n = Hn_split[1]
    Hn_effs = tangent_space_Hn(M, Hn_split, ks)
    Ntvs = length(Tvec_basis)
    Tvspins = map(spin, Tvec_basis)

    Hn_in_basis = zeros(T, (Ntvs,Ntvs))
    for j in 1:Ntvs
        ind = findfirst(ks, Tvspins[j])
        if ind > 0
            Bj = tvec_tensor(Tvec_basis[j])
            for k in 1:Ntvs
                if Tvspins[k] == Tvspins[j] + n
                    Bk = tvec_tensor(Tvec_basis[k])
                    Hn_in_basis[k,j] = dot(vec(Bk), Hn_effs[ind] * vec(Bj))
                end
            end
        end
    end
    
    Hn_in_basis
end

#--------------

function ising_local_MPO{T}(::Type{T}, shift::Float64=0.0; hz::Float64=1.0)::MPO_open{T}
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    h1 = zeros(Float64, 2,2,1,2)
    h2 = zeros(Float64, 2,2,2,1)
    
    h1[:,:,1,1] = -hz*Z + shift*I
    h1[:,:,1,2] = -X
    
    h2[:,:,1,1] = eye(2)
    h2[:,:,2,1] = X
    
    h1 = permutedims(h1, (3,2,4,1)) #[m1,ket,m2,bra]
    h2 = permutedims(h2, (3,2,4,1)) #[m1,ket,m2,bra]
    
    h1 = convert(MPOTensor{T}, h1)
    h2 = convert(MPOTensor{T}, h2)
    
    MPOTensor{T}[h1, h2]
end

function ising_PBC_MPO{T}(::Type{T}; hz::Float64=1.0)::MPO_PBC_uniform{T}
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,3,3)
    hB = zeros(Float64, 2,2,3,3)
    
    hM[:,:,1,1] = eye(2)
    hM[:,:,2,1] = X
    hM[:,:,3,1] = -hz*Z
    hM[:,:,3,2] = -X
    hM[:,:,3,3] = eye(2)
    
    hB[:,:,1,:] = hM[:,:,3,:]
    hB[:,:,2,3] = X
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hB = permutedims(hB, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hB = convert(MPOTensor{T}, hB)
    
    (hB, hM)
end

function ising_OBC_MPO{T}(::Type{T}; hz::Float64=1.0)::MPO_open_uniform{T}
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,3,3)
    
    hM[:,:,1,1] = eye(2)
    hM[:,:,2,1] = X
    hM[:,:,3,1] = -hz*Z
    hM[:,:,3,2] = -X
    hM[:,:,3,3] = eye(2)
    
    hL = zeros(Float64, 2,2,1,3)
    hR = zeros(Float64, 2,2,3,1)
    
    hL[:,:,1,:] = hM[:,:,3,:]
    hR[:,:,:,1] = hM[:,:,:,1]
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hL = permutedims(hL, (3,2,4,1)) #[m1,ket,m2,bra]
    hR = permutedims(hR, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hL = convert(MPOTensor{T}, hL)
    hR = convert(MPOTensor{T}, hR)
    
    (hL, hM, hR)
end

function ising_PBC_MPO_split{T}(::Type{T}; hz::Float64=1.0)::MPO_PBC_split{T}
    X = [0.0 1.0; 1.0 0.0]
    hL = reshape(-X.', (2,2,1,1))
    hR = reshape(X.', (2,2,1,1))
    hL = convert(MPOTensor{T}, permutedims(hL, (3,2,4,1)))
    hR = convert(MPOTensor{T}, permutedims(hR, (3,2,4,1)))
    
    h_B = MPOTensor{T}[hL, hR]
    
    (ising_OBC_MPO(T, hz=hz), h_B)
end

function ising_Hn_MPO_split{T}(::Type{T}, n::Int, N::Int; hz::Float64=1.0)
    (hL, hM, hR), (hb1, hb2) = ising_PBC_MPO_split(T, hz=hz)
    
    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,2,:] *= cis(n*1.5*2π/N)

    hR[3,:,1,:] *= cis(n*N*2π/N) #I realise this is not strictly necessary, but it shows intent! :)

    get_hM = (j::Int)->begin
        hM_j = copy(hM)
        hM_j[3,:,1,:] *= cis(n*j*2π/N)
        hM_j[3,:,2,:] *= cis(n*(j+0.5)*2π/N)
        hM_j
    end

    Hn_OBC = MPOTensor{T}[hL, (get_hM(n) for n in 2:N-1)..., hR]

    hb1 *= cis(n*(N+0.5)*2π/N)
    
    Hn_b = MPOTensor{T}[hb1, hb2]

    n, Hn_OBC, Hn_b
end

"""
-(lamda*X1*X2 + delta1*X1*X3 + delta2*Z1*Z2 + hz*Z1)
"""
function ANNNI_local_MPO{T}(::Type{T}; hz::Float64=1.0, delta1::Float64=0.0, delta2::Float64=0.0, lambda::Float64=1.0)::MPO_open{T}
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    h1 = zeros(Float64, 2,2,1,2)
    h2 = zeros(Float64, 2,2,2,2)
    h3 = zeros(Float64, 2,2,2,1)
    
    h1[:,:,1,1] = -X
    h1[:,:,1,2] = -Z
    
    h2[:,:,1,1] = lambda*X
    h2[:,:,1,2] = eye(2)
    h2[:,:,2,1] = delta2 * Z + I*hz
    
    h3[:,:,1,1] = eye(2)
    h3[:,:,2,1] = delta1 * X
    
    h1 = permutedims(h1, (3,2,4,1)) #[m1,ket,m2,bra]
    h2 = permutedims(h2, (3,2,4,1)) #[m1,ket,m2,bra]
    h3 = permutedims(h3, (3,2,4,1)) #[m1,ket,m2,bra]
    
    h1 = convert(MPOTensor{T}, h1)
    h2 = convert(MPOTensor{T}, h2)
    h3 = convert(MPOTensor{T}, h3)
    
    MPOTensor{T}[h1, h2, h3]
end

#FIXME: Actually, we would need two boundary tensors for this!
function ANNNI_PBC_MPO{T}(::Type{T}; hz::Float64=1.0, delta1::Float64=0.0, delta2::Float64=0.0, lambda::Float64=1.0)::MPO_PBC_uniform{T}
    @assert false "This needs fixing!"
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,5,5)
    hB = zeros(Float64, 2,2,5,5)
    
    hM[:,:,1,1] = eye(2)
    hM[:,:,2,1] = X
    hM[:,:,3,1] = Z
    hM[:,:,4,2] = eye(2)
    hM[:,:,5,2] = -lambda*X
    hM[:,:,5,3] = -delta2*Z-hz*I
    hM[:,:,5,4] = -delta1*X
    hM[:,:,5,5] = eye(2)
    
    hB[:,:,1,:] = hM[:,:,5,:]
    hB[:,:,:,5] = hM[:,:,:,1]
    hB[:,:,4,2] = eye(2)
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hB = permutedims(hB, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hB = convert(MPOTensor{T}, hB)
    
    (hB, hM)
end

function ANNNI_OBC_MPO{T}(::Type{T}; hz::Float64=1.0, delta1::Float64=0.0, delta2::Float64=0.0, lambda::Float64=1.0)::MPO_open_uniform{T}
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,5,5)
    
    hM[:,:,1,1] = eye(2)
    hM[:,:,2,1] = X
    hM[:,:,3,1] = Z
    hM[:,:,4,2] = eye(2)
    hM[:,:,5,2] = -lambda*X
    hM[:,:,5,3] = -delta2*Z-hz*I
    hM[:,:,5,4] = -delta1*X
    hM[:,:,5,5] = eye(2)
    
    hL = zeros(Float64, 2,2,1,5)
    hR = zeros(Float64, 2,2,5,1)
    
    hL[:,:,1,:] = hM[:,:,5,:]
    hL[:,:,1,1] = -hz*Z
    hR[:,:,:,1] = hM[:,:,:,1]
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hL = permutedims(hL, (3,2,4,1)) #[m1,ket,m2,bra]
    hR = permutedims(hR, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hL = convert(MPOTensor{T}, hL)
    hR = convert(MPOTensor{T}, hR)
    
    (hL, hM, hR)
end

function ANNNI_PBC_MPO_split{T}(::Type{T}; hz::Float64=1.0, delta1::Float64=0.0, delta2::Float64=0.0, lambda::Float64=1.0)::MPO_PBC_split{T}
    hL, hM, hR = ANNNI_OBC_MPO(T, hz=0.0, delta1=delta1, delta2=delta2, lambda=lambda)
    
    #Pick out only the needed terms from the OBC Hamiltonian. Reduces the max. bond dimension to 3.
    hL = hL[:,:,4:5,:]
    hM1 = hM[4:5,:,2:4,:]
    hM2 = hM[2:4,:,1:2,:]
    hR = hR[1:2,:,:,:]
    
    h_B = MPOTensor{T}[hL, hM1, hM2, hR]
    
    (ANNNI_OBC_MPO(T, hz=hz, delta1=delta1, delta2=delta2, lambda=lambda), h_B)
end

function weylops(p::Int)
    om = cis(2π / p)
    U = diagm(Complex128[om^j for j in 0:p-1])
    V = diagm(ones(p - 1), 1)
    V[end, 1] = 1
    U, V, om
end

function potts3_OBC_MPO{T}(::Type{T}; h::Float64=1.0)
    U, V, om = weylops(3)
    
    hM = zeros(Complex128, 3,3,4,4)

    hM[:,:,1,1] = eye(3)
    hM[:,:,2,1] = U
    hM[:,:,3,1] = U'
    hM[:,:,4,1] = -h/2 * (V+V')
    hM[:,:,4,2] = -0.5*U'
    hM[:,:,4,3] = -0.5*U
    hM[:,:,4,4] = eye(3)
    
    hL = zeros(Complex128, 3,3,1,4)
    hR = zeros(Complex128, 3,3,4,1)
    
    hL[:,:,1,:] = hM[:,:,4,:]
    hR[:,:,:,1] = hM[:,:,:,1]
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hL = permutedims(hL, (3,2,4,1)) #[m1,ket,m2,bra]
    hR = permutedims(hR, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hL = convert(MPOTensor{T}, hL)
    hR = convert(MPOTensor{T}, hR)
    
    (hL, hM, hR)
end

function potts3_local_MPO{T}(::Type{T}; h::Float64=1.0)
    hL, hM, hR = potts3_OBC_MPO(T, h=h)
    
    MPOTensor{T}[hL[:,:,1:3,:], hR[1:3,:,:,:]]
end

function potts3_PBC_MPO_split{T}(::Type{T}; h::Float64=1.0)::MPO_PBC_split{T}
    hL, hM, hR = potts3_OBC_MPO(T, h=h)
    
    h_B = MPOTensor{T}[hL[:,:,2:3,:], hR[2:3,:,:,:]]
    
    ((hL, hM, hR), h_B)
end

end
