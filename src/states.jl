"""
    MPO_PBC_uniform{T}

A Matrix Product Operator (MPO) with one translation-invariant middle (bulk) tensor and one boundary tensor.
"""
MPO_PBC_uniform{T} = Tuple{MPOTensor{T},MPOTensor{T}}

"""
    MPO_open_uniform{T}

The left, middle (translation-invariant bulk), and right tensors of an MPO with open boundary conditions.
"""
MPO_open_uniform{T} = Tuple{MPOTensor{T},MPOTensor{T},MPOTensor{T}} 

"""
    MPO_PBC_uniform_split{T}

An MPO for a circular system, split into an open uniform MPO for the bulk and a generic open MPO for the boundary.
"""
MPO_PBC_uniform_split{T} = Tuple{MPO_open_uniform{T}, MPO_open{T}}

"""
    MPO_PBC_split{T}

An MPO for a circular system, split into an generic open MPO for the bulk and a generic open MPO for the boundary.
"""
MPO_PBC_split{T} = Tuple{MPO_open{T}, MPO_open{T}}


"""
    A uniform Matrix Product State with periodic boundary conditions. 
    We need only store the MPS tensor `A`, which is the same for every site,
    and the number of sites for which `A` is intended to be used.
"""
mutable struct puMPState{T}
    A::MPSTensor{T}
    N::Int #number of sites
end

#Generates a random `puMPState` in canonical form
rand_puMPState(::Type{T}, d::Integer, D::Integer, N::Int) where {T} = puMPState(rand_MPSTensor_unitary(T, d, D), N)::puMPState{T}

Base.copy(M::puMPState) = puMPState(copy(M.A), M.N)

MPS.bond_dim(M::puMPState) = bond_dim(M.A)
MPS.phys_dim(M::puMPState) = phys_dim(M.A)
mps_tensor(M::puMPState) = M.A
num_sites(M::puMPState) = M.N

set_mps_tensor!(M::puMPState{T}, A::MPSTensor{T}) where {T} = M.A = A

"""
    Base.Vector(M::puMPState{T}) where T

Computes the full (dense) state vector from the (compressed) puMPS form.
Since puMPS typically represent state vectors that are far too big to fit
in memory, this should only be used for very small systems!
"""
function Base.Vector(M::puMPState{T}) where T
    N = num_sites(M)
    d = phys_dim(M)
    D = bond_dim(M)
    A = mps_tensor(M)

    N == 0 && return Vector{T}()

    res = A
    for j in 2:N
        @tensor res[l,p1,p2,r] := res[l,p1,ri] * A[ri,p2,r]
        res = reshape(res, (D, d^j, D))
    end
    @tensor res[p] := res[v,p,v]

    res
end

"""
    canonicalize_left!(M::puMPState; pinv_tol::Real=1e-12)

Modifies a puMPState in place via a gauge transformation to bring it into left-canonical form,
returning the puMPState and the diagonal matrices of what the Schmidt coefficients
would be if this were an infinite system.
"""
function canonicalize_left!(M::puMPState; pinv_tol::Real=1e-12)
    A = mps_tensor(M)
    
    dominant_ev, l, r = tm_dominant_eigs(A, A)
    
    lnew, rnew, x, xi = MPS.canonicalize_left(l, r)

    AL = gauge_transform(A, x, xi)
    set_mps_tensor!(M, AL)
    
    lambda = Diagonal(sqrt.(diag(rnew)))
    
    lambda_i = pinv(lambda, pinv_tol)
    
    M, lambda, lambda_i
end

MPS.canonicalize_left(M::puMPState) = canonicalize_left!(copy(M))

#Computes the one-site transfer matrix of a `puMPState` and returns it as an `MPS_MPO_TM` (see MPS).
MPS.TM_dense(M::puMPState) = TM_dense(mps_tensor(M), mps_tensor(M))

"""
    apply_blockTM_l(M::puMPState{T}, TM::MPS_MPO_TM{T}, N::Integer) where {T}

Applies `N` transfer matrices `TM_M` of the puMPState `M` 
to an existing transfer matrix `TM` by acting to the left:
```julia
TM * (TM_M)^N
```
"""
function apply_blockTM_l(M::puMPState{T}, TM::MPS_MPO_TM{T}, N::Integer) where {T}
    A = mps_tensor(M)
    work = workvec_applyTM_l(A, A)
    TMres = zero(TM)
    TM = N > 1 ? copy(TM) : TM #Never overwrite TM!
    for i in 1:N
        applyTM_l!(TMres, A, A, TM, work) #D^5 d
        TM, TMres = (TMres, TM)
    end
    TM
end

"""
    blockTM_dense(M::puMPState{T}, N::Integer) where {T}

Computes the `N`th power of the transfer matrix of the puMPState `M`. Time cost: O(`bond_dim(M)^5`).
"""
function blockTM_dense(M::puMPState{T}, N::Integer) where {T}
    #TODO: Depending on d vs. D, block the MPS tensors first to form an initial blockTM at cost D^4 d^blocksize.
    D = bond_dim(M)
    TM = N == 0 ? eyeTM(T, D) : apply_blockTM_l(M, TM_dense(M), N-1)
    TM
end

"""
    blockTMs(M::puMPState{T}, N::Integer=num_sites(M)) where {T}

Computes the powers of the transfer matrix of the puMPState `M` up to and including the `N`th power.
Time cost: O(`bond_dim(M)^5`).
"""
function blockTMs(M::puMPState{T}, N::Integer=num_sites(M)) where {T}
    A = mps_tensor(M)
    TMs = MPS_MPO_TM{T}[TM_dense(M)]
    work = workvec_applyTM_l(A, A)
    for n in 2:N
        TMres = similar(TMs[end])
        push!(TMs, applyTM_l!(TMres, A, A, TMs[end], work))
    end
    TMs
end

"""
    LinearAlgebra.norm(M::puMPState{T}; TM_N::MPS_MPO_TM{T}=blockTM_dense(M, num_sites(M))) where {T}

The norm of the puMPState `M`, optionally reusing the precomputed 
`N`th power of the transfer matrix `TM_N`, where `N = num_sites(M)`.
"""
function LinearAlgebra.norm(M::puMPState{T}; TM_N::MPS_MPO_TM{T}=blockTM_dense(M, num_sites(M))) where {T}
    sqrt(tr(TM_N))
end

"""
    LinearAlgebra.normalize!(M::puMPState{T}; TM_N::MPS_MPO_TM{T}=blockTM_dense(M, num_sites(M))) where {T}
    
Normalizes the puMPState `M` in place, optionally reusing the precomputed 
`N`th power of the transfer matrix `TM_N`, where `N = num_sites(M)`.
"""
function LinearAlgebra.normalize!(M::puMPState{T}; TM_N::MPS_MPO_TM{T}=blockTM_dense(M, num_sites(M))) where {T}
    rmul!(mps_tensor(M), T(1.0 / norm(M, TM_N=TM_N)^(1.0/num_sites(M))) )
    M
end
    
LinearAlgebra.normalize(M::puMPState{T}; TM_N::MPS_MPO_TM{T}=blockTM_dense(M, num_sites(M))) where {T} = normalize!(copy(M), TM_N=TM_N)

"""
    LinearAlgebra.normalize!(M::puMPState{T}, blkTMs::Vector{MPS_MPO_TM{T}}) where {T}

Normalizes the puMPState `M` in place together with a set of precomputed powers of the transfer matrix `blkTMs`.
"""
function LinearAlgebra.normalize!(M::puMPState{T}, blkTMs::Vector{MPS_MPO_TM{T}}) where {T}
    N = num_sites(M)
    normM = norm(M, TM_N=blkTMs[N])
    
    rmul!(mps_tensor(M), T(1.0 / normM^(1.0/N)) )
    for n in 1:N
        rmul!(blkTMs[n], T(1.0 / normM^(2.0/n)) ) 
    end
    M, blkTMs
end

"""
    LinearAlgebra.dot(M::puMPState, psi::AbstractVector)

Computes the inner product of a puMPState with a state vector (dense representation).
"""
function LinearAlgebra.dot(M::puMPState, psi::AbstractVector)
    N = num_sites(M)
    D = bond_dim(M)
    d = phys_dim(M)
    length(psi) == d^N || DimensionMismatch("")

    A = mps_tensor(M)

    psi_r = reshape(psi, (d, d^(N-1)))
    @tensor con[r,phys,l] := conj(A[l,p1,r]) * psi_r[p1,phys]
    for n=2:N
        con = reshape(con, (D, d, d^(N-n), D)) #r,p1,phys,l
        @tensor con[r,phys,l] := conj(A[li,p1,r]) * con[li,p1,phys,l]
    end
    con = reshape(con, (D,D))

    tr(con)
end

LinearAlgebra.dot(psi::AbstractVector, M::puMPState) = conj(dot(M, psi))

"""
    expect_nn(M::puMPState{Ts}, op::Array{Top,4}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_MPO_TM{Ts}}=MPS_MPO_TM{Ts}[]) where {Ts, Top}

Computes the expectation value with respect to `M` of a nearest-neighbour operator,
supplied as a 4-dimensional `Array` defined as
    `op[t1,t2, s1,s2]` = <t1,t2|op|s1,s2>
with each index enumerating the basis for the one-site physical Hilbert space
according to which the puMPState `M` is defined.

Optionally uses precomputed powers of the transfer matrix as `blkTMs`.
In case `MPS_is_normalized == false` computes the norm of `M` at the same time.
"""
function expect_nn(M::puMPState{Ts}, op::Array{Top,4}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_MPO_TM{Ts}}=MPS_MPO_TM{Ts}[]) where {Ts, Top}
    N = num_sites(M)
    A = mps_tensor(M)
    
    if MPS_is_normalized && length(blkTMs) < N-2
        TM = TM_dense_op_nn(A,A,A,A, op) #D^4 d^2
        TM = apply_blockTM_l(M, TM, N-2) #NOTE: If N-2 >> D it is cheaper to do full D^6 multiplication with a block.
        
        return tr(TM)
    else
        #We can use most of the block TM for both the norm and the expectation value.
        TM = length(blkTMs) >= N-2 ? blkTMs[N-2] : blockTM_dense(M, N-2)
        normsq = length(blkTMs) == N ? tr(blkTMs[N]) : tr(apply_blockTM_l(M, TM, 2))
        
        TM = applyTM_op_nn_l(A,A,A,A, op, TM)
        
        return tr(TM) / normsq
    end
end

"""
    expect(M::puMPState{T}, op::MPO_open{T}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_MPO_TM{T}}=MPS_MPO_TM{T}[]) where {T}

Computes the expectation value of an MPO. The MPO may have between 1 and `num_sites(M)` sites.
If it has the maximum number of sites, it is allowed to have open or periodic boundary conditions.
Otherwise the MPO bond dimension must go to 1 at both ends.
See MPS for the definition of `MPO_open`.

Optionally uses precomputed powers of the transfer matrix as `blkTMs`.
In case `MPS_is_normalized == false` computes the norm of `M` at the same time.
"""
function expect(M::puMPState{T}, op::MPO_open{T}; MPS_is_normalized::Bool=true, blkTMs::Vector{MPS_MPO_TM{T}}=MPS_MPO_TM{T}[]) where {T}
    N = num_sites(M)
    A = mps_tensor(M)
    D = bond_dim(M)
    
    Nop = length(op)
    
    if N == Nop
        TMop = TM_dense_MPO(M, op)
        res = tr(TMop)

        if !MPS_is_normalized
            normsq = length(blkTMs) == N ? tr(blkTMs[N]) : norm(M)^2
            res /= normsq
        end
    else
        TM = length(blkTMs) >= N-Nop ? blkTMs[N-Nop] : blockTM_dense(M, N-Nop)
        
        TMop = applyTM_MPO_l(M, op, TM)
        
        res = tr(TMop)
        
        if !MPS_is_normalized
            normsq = length(blkTMs) == N ? tr(blkTMs[N]) : tr(apply_blockTM_l(M, TM, Nop))
            res /= normsq
        end
    end
    
    res
end

"""
    expect(M::puMPState{T}, O::MPO_PBC_uniform{T}) where {T}

Computes the expectation value of a global MPO with periodic boundary conditions.
See MPS for the definition of `MPO_PBC_uniform`.
"""
function expect(M::puMPState{T}, O::MPO_PBC_uniform{T}) where {T}
    N = num_sites(M)
    OB, OM = O    
    O_full = MPOTensor{T}[OB, (OM for j in 2:N)...]
    expect(M, O_full)
end

"""
    MPS.TM_dense_MPO(M::puMPState{T}, O::MPO_open{T})::MPS_MPO_TM{T} where {T}

The transfer matrix for the entire length of the MPO `O`.
"""
function MPS.TM_dense_MPO(M::puMPState{T}, O::MPO_open{T})::MPS_MPO_TM{T} where {T}
    A = mps_tensor(M)
    applyTM_MPO_l(M, O[2:end], TM_dense_MPO(A, A, O[1]))
end   

"""
    blockTMs_MPO(M::puMPState{T}, O::MPO_PBC_uniform{T}, N::Integer=num_sites(M)) where {T}

Block transfer matrices for the MPO `O` for sites 1 to n, with `n in 1:N`.
These are not all powers of the same transfer matrix, since the MPO with PBC
is generally not completely uniform.
"""
function blockTMs_MPO(M::puMPState{T}, O::MPO_PBC_uniform{T}, N::Integer=num_sites(M)) where {T}
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
    MPS.applyTM_MPO_l(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}; 
        work::Vector{T}=Vector{T}())::MPS_MPO_TM{T} where {T}

Apply an MPS-MPO transfer matrix `TM1` derived from the puMPState `M` and the MPO `O` to an
exisiting MPS-MPO transfer matrix `TM2`, acting to the left: `TM2 * TM1`.

A vector for holding intermediate results may be supplied as `work`. It may be resized!
"""
function MPS.applyTM_MPO_l(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}; 
    work::Vector{T}=Vector{T}())::MPS_MPO_TM{T} where {T}

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
    MPS.res_applyTM_MPO_l(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}) where {T}

Prepare appropriately-sized result arrays `TMres` to hold the intermediate and final results of 
`applyTM_MPO_l!(TMres, M, O, TM2)`.
"""
function MPS.res_applyTM_MPO_l(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}) where {T}
    A = mps_tensor(M)
    res = MPS_MPO_TM{T}[]
    for n in 1:length(O)
        push!(res, res_applyTM_MPO_l(A, A, O[n], TM2))
        TM2 = res[end]
    end
    res
end

"""
    MPS.workvec_applyTM_MPO_l(M::puMPState{T}, O::MPO_open{T}, TM2s::Vector{MPS_MPO_TM{T}}) where {T}

Prepare working-memory vector for applyTM_MPO_l!().
"""
function MPS.workvec_applyTM_MPO_l(M::puMPState{T}, O::MPO_open{T}, TM2s::Vector{MPS_MPO_TM{T}}) where {T}
    A = mps_tensor(M)
    len = 0
    for j in 1:length(O)
        len = max(len, worklen_applyTM_MPO_l(A, A, O[j], TM2s[j]))
    end
    workMPO = Vector{T}(undef, len)
end

"""
    MPS.applyTM_MPO_l!(TMres::Vector{MPS_MPO_TM{T}}, M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T} where {T}

Apply an MPS-MPO transfer matrix `TM1` derived from the puMPState `M` and the MPO `O` to an
exisiting MPS-MPO transfer matrix `TM2`, acting to the left:
    `TM2 * TM1`
This version accepts a vector `work` for working memory as well as a preallocated
set of result arrays `TMres`.
"""
function MPS.applyTM_MPO_l!(TMres::Vector{MPS_MPO_TM{T}}, M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T}, work::Vector{T})::MPS_MPO_TM{T} where {T}
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
    MPS.applyTM_MPO_r(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T};
            work::Vector{T}=Vector{T}())::MPS_MPO_TM{T} where {T}

Apply an MPS-MPO transfer matrix `TM1` derived from the puMPState `M` and the MPO `O` to an
exisiting MPS-MPO transfer matrix `TM2`, acting to the right: `TM1 * TM2`.

A vector for holding intermediate results may be supplied as `work`. It may be resized!
"""
function MPS.applyTM_MPO_r(M::puMPState{T}, O::MPO_open{T}, TM2::MPS_MPO_TM{T};
    work::Vector{T}=Vector{T}())::MPS_MPO_TM{T} where {T}
    A = mps_tensor(M)
    
    TM = TM2
    if length(O) > 0
        TMres = res_applyTM_MPO_r(A, A, O[end], TM)
        for n in length(O):-1:1
            TMres = size(O[n],1) != size(O[n], 3) ? res_applyTM_MPO_r(A, A, O[n], TM) : TMres
            workvec_applyTM_MPO_r!(work, A, A, O[n], TM)
            TM = applyTM_MPO_r!(TMres, A, A, O[n], TM, work)
        end
    end
    TM
end

"""
    derivatives_1s(M::puMPState{T}, h::MPO_open{T}; blkTMs::Vector{MPS_MPO_TM{T}}=blockTMs(M, num_sites(M)-1), e0::Real=0.0) where {T}

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
function derivatives_1s(M::puMPState{T}, h::MPO_open{T}; blkTMs::Vector{MPS_MPO_TM{T}}=blockTMs(M, num_sites(M)-1), e0::Real=0.0) where {T}
    A = mps_tensor(M)
    N = num_sites(M)
    D = bond_dim(M)
    
    e0 = real(T)(e0) #will be used for scaling, so need it in the working precision

    j = 1
    TM = blkTMs[j]
    
    #Transfer matrix with one H term
    TM_H = TM_dense_MPO(M, h)
    
    #Subtract energy density e0 * I.
    #Note: We do this for each h term individually in order to avoid a larger subtraction later.
    #Assumption: This is similarly accurate to subtracting I*e0 from the Hamiltonian itself.
    #The transfer matrices typically have similar norm before and after subtraction, even when
    #the final gradient has small physical norm.
    axpy!(-e0, blkTMs[length(h)], TM_H) 
    
    TM_H_res = similar(TM_H)    
    
    work = workvec_applyTM_l(A, A)
    
    TMMPO_res = res_applyTM_MPO_l(M, h, TM)
    workMPO = workvec_applyTM_MPO_l(M, h, vcat(MPS_MPO_TM{T}[TM], TMMPO_res[1:end-1]))
    
    for k in length(h)+1:N-1 #leave out one site (where we take the derivative)
        #Extend TM_H
        applyTM_l!(TM_H_res, A, A, TM_H, work)
        TM_H, TM_H_res = (TM_H_res, TM_H)
        
        #New H term
        TM_H_add = applyTM_MPO_l!(TMMPO_res, M, h, TM, workMPO)
        axpy!(-e0, blkTMs[j+length(h)], TM_H_add) #Subtract energy density e0 * I
        
        j += 1
        TM = blkTMs[j]
        
        axpy!(real(T)(1.0), TM_H_add, TM_H) #add new H term to TM_H
    end
    
    #effective ham terms that do not act on gradient site
    axpy!(-length(h)*e0, blkTMs[N-1], TM_H) #Subtract energy density for the final terms

    #Add only the A, leaving a conjugate gap.
    TM_H_r = TM_convert(TM_H)
    @tensor d_A[l, s, r] := A[k1, s, k2] * TM_H_r[k2,r, k1,l]
    
    #NOTE: TM now has N-length(h) sites
    for n in 1:length(h)
        TM_H = applyTM_MPO_l(M, h[1:n-1], TM, work=workMPO)
        TM_H = applyTM_MPO_r(M, h[n+1:end], TM_H, work=workMPO)
        hn = h[n]
        @tensor d_A[l, t, r] += (A[k1, s, k2] * TM_H[k2,m2,r, k1,m1,l]) * hn[m1,s,m2,t] #allocates temporaries
    end
    
    d_A
end

function eff_Ham_1s(M::puMPState{T}, h::MPO_open{T}; blkTMs::Vector{MPS_MPO_TM{T}}=blockTMs(M, num_sites(M)-1), e0::Real=0.0) where {T}
    A = mps_tensor(M)
    N = num_sites(M)
    D = bond_dim(M)

    e0 = real(T)(e0) #will be used for scaling, so need it in the working precision
    
    j = 1
    TM = blkTMs[j]
    
    #Transfer matrix with one H term
    TM_H = TM_dense_MPO(M, h)
    
    #Subtract energy density e0 * I.
    #Note: We do this for each h term individually in order to avoid a larger subtraction later.
    #Assumption: This is similarly accurate to subtracting I*e0 from the Hamiltonian itself.
    #The transfer matrices typically have similar norm before and after subtraction, even when
    #the final gradient has small physical norm.
    axpy!(-e0, blkTMs[length(h)], TM_H) 
    
    TM_H_res = similar(TM_H)    
    
    work = workvec_applyTM_l(A, A)
    
    TMMPO_res = res_applyTM_MPO_l(M, h, TM)
    workMPO = workvec_applyTM_MPO_l(M, h, vcat(MPS_MPO_TM{T}[TM], TMMPO_res[1:end-1]))
    
    for k in length(h)+1:N-1 #leave out one site (where we take the derivative)
        #Extend TM_H
        applyTM_l!(TM_H_res, A, A, TM_H, work)
        TM_H, TM_H_res = (TM_H_res, TM_H)
        
        #New H term
        TM_H_add = applyTM_MPO_l!(TMMPO_res, M, h, TM, workMPO)
        axpy!(-e0, blkTMs[j+length(h)], TM_H_add) #Subtract energy density e0 * I
        
        j += 1
        TM = blkTMs[j]
        
        axpy!(real(T)(1.0), TM_H_add, TM_H) #add new H term to TM_H
    end
    
    #effective ham terms that do not act on gradient site
    axpy!(-length(h)*e0, blkTMs[N-1], TM_H) #Subtract energy density for the final terms

    d = phys_dim(M)
    e = Matrix{T}(I,d,d)
    TM_H_r = TM_convert(TM_H)
    @tensor Heff[Vb1, Pb, Vb2, Vt1, Pt, Vt2] := TM_H_r[Vt2,Vb2, Vt1,Vb1] * e[Pt, Pb]
    
    #NOTE: TM now has N-length(h) sites
    for n in 1:length(h)
        TM_H = applyTM_MPO_l(M, h[1:n-1], TM, work=workMPO)
        TM_H = applyTM_MPO_r(M, h[n+1:end], TM_H, work=workMPO)
        hn = h[n]
        @tensor Heff[Vb1, Pb, Vb2, Vt1, Pt, Vt2] += TM_H[Vt2,m2,Vb2, Vt1,m1,Vb1] * hn[m1,Pt,m2,Pb]
    end
    
    Heff
end

function eff_Hams_Ac_C(M::puMPState{T}, lambda_i::AbstractMatrix{T}, Heff::MPS_MPO_TM{T}, blkTM_Nm1::MPS_MPO_TM{T}) where {T}
    N = num_sites(M)
    D = bond_dim(M)
    d = phys_dim(M)

    lambda_i = Matrix(lambda_i)

    blkTM_Nm1 = TM_convert(blkTM_Nm1) #there is no MPO content

    #Heff is for the uniform tensors in M (usual in left canonical form)
    @tensor Heff_Ac[V1b,Pb,V2b, V1t,Pt,V2t] := lambda_i[vb,V2b] * (Heff[V1b,Pb,vb, V1t,Pt,vt] * lambda_i[vt,V2t])

    e = Matrix{T}(I,d,d)
    @tensor N_Ac_noe[V1b,V2b, V1t,V2t] := (lambda_i[vb,V2b] * (blkTM_Nm1[vt,vb, V1t,V1b] * lambda_i[vt,V2t]))
    @tensor N_Ac[V1b,Pb,V2b, V1t,Pt,V2t] := N_Ac_noe[V1b,V2b, V1t,V2t] * e[Pb,Pt]

    A = mps_tensor(M)
    @tensor Heff_C[V1b,V2b, V1t,V2t] := A[vt,pt,V1t] * conj(A[vb,pb,V1b]) * Heff_Ac[vb,pb,V2b, vt,pt,V2t]
    @tensor N_C[V1b,V2b, V1t,V2t] := A[vt,p,V1t] * conj(A[vb,p,V1b]) * N_Ac_noe[vb,V2b, vt,V2t]

    Heff_Ac, N_Ac, Heff_C, N_C
end

function vumps_local_gnd(X::Array{T,N}, Heff::Array{T,M}, Nmat::Array{T,M}, tol::Real; ncv=20) where {T,N,M}
    @assert M == 2N
    Heff = reshape(Heff, (prod(size(Heff)[1:N]), prod(size(Heff)[N+1:2N])))
    Nmat = reshape(Nmat, (prod(size(Nmat)[1:N]), prod(size(Nmat)[N+1:2N])))

    # KrylovKit does not currently support generalized eigenvalue problems directly,
    # so we do a full pseudo-inverse here at cost ~D^6 :(
    # Could improve this for N_Ac by inverting the components individually!
    @time Ninv = pinv(Nmat, tol)
    @time ev, eV, info = eigsolve(Ninv*Heff, vec(X), 1, :SR, tol=tol, krylovdim=ncv)
    info.converged == 0 && @warn "Eigenvector did not converge after $(info.numiter) iterations."

    @show ev
    reshape(eV[1], size(X))
end

function vumps_update_state(Ac::MPSTensor{T}, C::Matrix{T}) where {T}
    d = phys_dim(Ac)
    D = bond_dim(Ac)
    Ul,sl,Vl = svd(reshape(Ac, (d*D, D)) * C')
    Al = reshape(Ul * Vl', size(Ac))
    
    Ur,sr,Vr = svd(C' * reshape(Ac, (D, d*D)))
    Ar = reshape(Ur * Vr', size(Ac))
    
    el = norm(reshape(Ac, (d*D, D)) - reshape(Al, (d*D, D)) * C)
    er = norm(reshape(Ac, (D, d*D)) - C * reshape(Ar, (D, d*D)))
    
    Al, Ar, el, er
end

function vumps_opt!(M::puMPState{T}, hMPO::MPO_open{T}, tol::Real; maxitr::Integer=100, ncv=20) where {T}
    N = num_sites(M)
    blkTMs = blockTMs(M)
    normalize!(M, blkTMs)
    En = real(expect(M, hMPO, blkTMs=blkTMs))
    
    stol = 1e-12
    Ac_normgrad = Inf

    @time M, C, Ci = canonicalize_left!(M)
    C = Matrix(C)
    Ci = Matrix(Ci)
    Al = mps_tensor(M)
    @tensor Ac[V1, P, V2] := Al[V1, P, v] * C[v, V2] 
    
    for k in 1:maxitr
        println("Itr: $k")
        
        blkTMs = blockTMs(M)
        En_prev = En
        En = real(expect(M, hMPO, blkTMs=blkTMs))
        Heff_Al = eff_Ham_1s(M, hMPO, blkTMs=blkTMs, e0=En)
        Heff_Ac, N_Ac, Heff_C, N_C = eff_Hams_Ac_C(M, Ci, Heff_Al, blkTMs[N-1])

        @tensor Ac_grad[V1,P,V2] := Heff_Ac[V1,P,V2, v1,p,v2] * Ac[v1,p,v2]
        Ac_normgrad = norm(Ac_grad)

        @tensor C_grad[V1,V2] := Heff_C[V1,V2, v1,v2] * C[v1,v2]
        C_normgrad = norm(C_grad)

        Ac_new = vumps_local_gnd(Ac, Heff_Ac, N_Ac, stol, ncv=ncv)
        C_new = vumps_local_gnd(C, Heff_C, N_C, stol, ncv=ncv)

        Al, Ar, el, er = vumps_update_state(Ac_new, C_new)
        @show el

        set_mps_tensor!(M, Al)
        normalize!(M)
        M, C, Ci = canonicalize_left!(M)
        @show abs(C[1,1])
        C = Matrix(C)
        Ci = Matrix(Ci)

        #rmul!(Al, 1.0 / nrm^(1.0/num_sites(M)))
        #set_mps_tensor!(M, Al)
        #FIXME: Probably need to scale C_new
        #normalize!(C_new)
        #C = C_new
        #Ci = pinv(C, 1e-12)
        @tensor Ac[V1, P, V2] = Al[V1, P, v] * C[v, V2] 

        println("$Ac_normgrad, $C_normgrad, $En, $(En-En_prev)")
        if Ac_normgrad < tol
            break
        end
    end

    M, Ac_normgrad
end

function BiCGstab(M,V,X0,tol::Real; max_itr::Integer=100)
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
    
    !converged && @warn "BiCGStab did not converge (tol: $tol, max_itr: $max_itr)"
    
    x
end

# try
#     @pyimport scipy.sparse.linalg as SLA
# catch
#     warn("Could not import sparse linear algebra from Scipy.")
#     SLA = nothing
# end

# function BiCGstab_scipy(M,V,X0,tol::Real; max_itr::Integer=100, max_attempts::Integer=1)
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

# function lGMRes_scipy(M,V,X0,tol::Real; max_itr::Integer=100)
#     res, info = SLA.lgmres(M, V, X0, tol, maxiter=max_itr)
#     info < 0 && error("lGMRes failed due to illegal input or breakdown")
#     info > 0 && warn("lGMRes did not converge (tol: $tol, max_itr: $max_itr)")
#     res
# end

"""
    gradient_central(M::puMPState{T}, inv_lambda::AbstractMatrix{T}, d_A::MPSTensor{T}; 
        sparse_inverse::Bool=true, pinv_tol::Real=1e-12,
        max_itr::Integer=500, tol::Real=1e-12,
        grad_Ac_init::MPSTensor{T}=rand_MPSTensor(T, phys_dim(M), bond_dim(M)),
        blkTMs::Vector{MPS_MPO_TM{T}}=MPS_MPO_TM{T}[]) where {T}

Converts the energy derivatives supplied by `derivatives_1s` into the energy gradient for
a single tensor of the puMPState `M`.

We first do a gauge transformation using `inv_lambda` to bring one tensor into the centre gauge.
This makes the inverse of the induced physical metric `Nc` on the one-site tensor parameters, which
is needed to compute the gradient from the derivatives, better conditioned.

If `sparse_inverse == false`, the inverse is computed explicitly as a pseudo inverse at cost O(`bond_dim(M)^6`).
Otherwise it is computed implicitly using the BiCGStab solver at cost O(`bond_dim(M)^4 * num_iter`).

The physical norm of the gradient is also computed and returned.
"""
function gradient_central(M::puMPState{T}, inv_lambda::AbstractMatrix{T}, d_A::MPSTensor{T}; 
        sparse_inverse::Bool=true, pinv_tol::Real=1e-12,
        max_itr::Integer=500, tol::Real=1e-12,
        grad_Ac_init::MPSTensor{T}=rand_MPSTensor(T, phys_dim(M), bond_dim(M)),
        blkTMs::Vector{MPS_MPO_TM{T}}=MPS_MPO_TM{T}[]) where {T}
    N = num_sites(M)
    D = bond_dim(M)
    d = phys_dim(M)
    
    inv_lambda = Matrix(inv_lambda)
    
    T1 = TM_convert(length(blkTMs) >= N-1 ? blkTMs[N-1] : blockTM_dense(M, N-1))
    
    #Overlap matrix in central gauge (except for the identity on the physical dimension)
    @tensor Nc[b2,b1,t2,t1] := inv_lambda[t1,ti] * inv_lambda[b1,bi] * T1[ti,bi,t2,b2]
    Nc = reshape(Nc, (D^2, D^2))
    ## Note that above can also be obtained from the normalization process
    
    @tensor d_Ac[v1,v2,s] := d_A[v1,s,vi] * inv_lambda[vi,v2] # now size (D,D,d)
    
    grad_Ac = zero(d_Ac)
    
    if sparse_inverse
        grad_Ac_init = tensorcopy(grad_Ac_init, [:a,:b,:c], [:a,:c,:b]) # now size (D,D,d)
        #Split the inverse problem along the physical dimension, since N acts trivially on that factor. Avoids constructing N x I.
        for s in 1:d
            grad_vec = BiCGstab(Nc, vec(view(d_Ac, :,:,s)), vec(view(grad_Ac_init, :,:,s)), tol, max_itr=max_itr)
            copyto!(view(grad_Ac, :,:,s), grad_vec)
        end
    else
        #Dense version
        #Nc_i = inv(Nc)
        Nc_i = pinv(Nc, pinv_tol)
        for s in 1:d
            grad_vec = Nc_i * vec(view(d_Ac, :,:,s))
            copyto!(view(grad_Ac, :,:,s), grad_vec)
        end
    end
    
    @tensor grad_A[v1,s,v2] := grad_Ac[v1,vi,s] * inv_lambda[vi,v2] # back to (D,d,D)
    
    norm_grad_A = sqrt(abs(dot(vec(grad_A), vec(d_A))))
    
    grad_A, norm_grad_A, tensorcopy(grad_Ac, [:a,:b,:c], [:a,:c,:b])
end

struct EnergyHighException{T<:Real} <: Exception
    stp::T
    En::T
end
struct WolfeAbortException{T<:Real} <: Exception 
    stp::T
    En::T
end

"""
    line_search_energy(M::puMPState{T}, En0::Real, grad::MPSTensor{T}, grad_normsq::Real, step::Real, hMPO::MPO_open{T}; 
        itr::Integer=10, rel_tol::Real=1e-1, max_attempts::Integer=3, wolfe_c1::Real=100.0
        ) where {T}

Conducts a line search starting at the puMPState `M` to find the puMPState closest to the energetic minimum along 
the search-direction specified by `grad`.

`En0` should contain the energy-density of `M`, which will be used as a reference point: 
Steps that increase the energy are avoided, although not completely excluded. 
Where they occur, they will typically be small compared to `step`.

`step` is a guide for the initial step length.

`hMPO` is the local Hamiltonian term in MPO form. It is used to compute the energy density.
"""
function line_search_energy(M::puMPState{T}, En0::Real, grad::MPSTensor{T}, grad_normsq::Real, step::Real, hMPO::MPO_open{T}; 
        itr::Integer=10, rel_tol::Real=1e-1, max_attempts::Integer=3, wolfe_c1::Real=100.0
    ) where {T}
    M_new = copy(M)
    num_calls::Int = 0
    attempt::Int = 0
    
    f = (stp::Float64)->begin
        num_calls += 1
        
        set_mps_tensor!(M_new, mps_tensor(M) .- real(T)(stp) .* grad)
        
        En = real(expect(M_new, hMPO, MPS_is_normalized=false)) #computes the norm and energy-density in one step
        
        #println("Linesearch: $stp, $En")

        #Abort the search if the first step already increases the energy compared to the initial state
        num_calls == 1 && En > En0 && throw(EnergyHighException{Float64}(stp, Float64(En)))
        
        #Note: This is the first Wolfe condition, plus a minimum step size, since we don't want to compute the gradient...
        #Probably it effectively only serves to reduce the maximum step size reached, thus we turn it off by setting wolfe_c1=100.
        stp > 1e-2 && En <= En0 - wolfe_c1 * stp * grad_normsq && throw(WolfeAbortException{Float64}(stp, En))
        
        En
    end
    
    step = Float64(step)

    res = nothing
    while attempt <= max_attempts
        try
            attempt += 1
            ores = optimize(f, step/5, step*1.8, Brent(), iterations=itr, rel_tol=Float64(rel_tol), store_trace=false, extended_trace=false)
            res = Optim.minimizer(ores), Optim.minimum(ores)
            break
        catch e
            if isa(e, EnergyHighException)
                if attempt < max_attempts
                    @warn "Linesearch: Initial step was too large. Adjusting!"
                    step *= 0.1
                    num_calls = 0
                else
                    @warn "Linesearch: Initial step was too large. Aborting!"
                    res = e.stp, e.En
                    break
                end
            elseif isa(e, WolfeAbortException)
                @info "Linesearch: Early stop due to good enough step!"
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
    minimize_energy_local!(M::puMPState{T}, hMPO::MPO_open{T}, maxitr::Integer;
        tol::Real=1e-6,
        step::Real=0.001,
        grad_max_itr::Integer=500,
        grad_sparse_inverse::Bool=false,
        use_phys_grad::Bool=true) where {T}

Optimises the puMPState `M` to minimize the energy with respect to a translation-invariant local Hamiltonian.
The local Hamiltonian term is supplied as an open MPO `hMPO`, which is a vector of `MPOTensor`:
`hMPO = MPOTensor[h1,h2,...,hn]`.

This MPO has a range of `n` sites. The `MPOTensor`s `h1` and `hn` must have outer MPO bond dimension 1.
For a nearest-neighbour Hamiltonian, `n=2`.
"""
function minimize_energy_local!(M::puMPState{T}, hMPO::MPO_open{T}, maxitr::Integer;
        tol::Real=1e-6,
        step::Real=0.001,
        grad_max_itr::Integer=500,
        grad_sparse_inverse::Bool=false,
        use_phys_grad::Bool=true) where {T}
    blkTMs = blockTMs(M)
    normalize!(M, blkTMs)
    En = real(expect(M, hMPO, blkTMs=blkTMs))
    
    grad_Ac = rand_MPSTensor(T, phys_dim(M), bond_dim(M)) #Used to initialise the BiCG solver
    stol = 1e-12
    norm_grad = Inf
    
    for k in 1:maxitr
        M, lambda, lambda_i = canonicalize_left!(M)
        
        blkTMs = blockTMs(M)
        deriv = derivatives_1s(M, hMPO, blkTMs=blkTMs, e0=En)

        if use_phys_grad
            grad, norm_grad, grad_Ac = gradient_central(M, lambda_i, deriv, sparse_inverse=grad_sparse_inverse, 
                                                  grad_Ac_init=grad_Ac, blkTMs=blkTMs, tol=stol, max_itr=grad_max_itr)
        else
            grad = deriv
            bTM_Nm1 = TM_convert(blkTMs[num_sites(M)-1])
            @tensor ng2[] := deriv[vt1, p, vt2] * (conj(deriv[vb1, p, vb2]) * bTM_Nm1[vt2,vb2, vt1,vb1])
            norm_grad = sqrt(real(scalar(ng2)))
        end
        
        if norm_grad < tol
            break
        end

        stol = min(1e-6, max(norm_grad^2/10, 1e-12))
        En_prev = En

        step_corr = min(max(step, 0.001),0.1)
        step, En = line_search_energy(M, En, grad, norm_grad^2, step_corr, hMPO)
        
        println("$k, $norm_grad, $step, $En, $(En-En_prev)")

        Anew = mps_tensor(M) .- real(T)(step) .* grad
        set_mps_tensor!(M, Anew)
        normalize!(M)
    end
    
    normalize!(M)
    M, norm_grad
end

function minimize_energy_local_CG!(M::puMPState{T}, hMPO::MPO_open{T}, maxitr::Integer;
    tol::Real=1e-6,
    step::Real=0.01,
    cg_steps_max::Integer=10,
    use_phys_grad::Bool=true) where {T}

    blkTMs = blockTMs(M)
    normalize!(M, blkTMs)
    En = real(expect(M, hMPO, blkTMs=blkTMs))
    En_prev = En

    grad = nothing
    step_dir = nothing
    beta = 0.0
    norm_grad = Inf

    step_internal = step

    cg_steps = 0

    ts = fill!(Vector{Float64}(undef, maxitr), NaN)
    ens = fill!(similar(ts), NaN)
    ngs = fill!(similar(ts), NaN)
    steps = fill!(similar(ts), NaN)
    betas = fill!(similar(ts), NaN)

    tic()
    for k in 1:maxitr
        if step_internal == 0
            beta = 0.0
        else
            M, lambda, lambda_i = canonicalize_left!(M)
            
            blkTMs = blockTMs(M)
            deriv = derivatives_1s(M, hMPO, blkTMs=blkTMs, e0=En)

            norm_grad_prev = norm_grad
            if use_phys_grad
                grad, norm_grad, grad_Ac = gradient_central(M, lambda_i, deriv, sparse_inverse=false, 
                                                                blkTMs=blkTMs)
            else
                grad = deriv
                bTM_Nm1 = TM_convert(blkTMs[num_sites(M)-1])
                @tensor ng2[] := deriv[vt1, p, vt2] * (conj(deriv[vb1, p, vb2]) * bTM_Nm1[vt2,vb2, vt1,vb1])
                norm_grad = sqrt(real(scalar(ng2)))
            end

            beta = norm_grad^2 / norm_grad_prev^2
        end
        
        if beta > 100
            cg_steps_max > 1 && @warn "Very large beta=$(beta), resetting CG after $cg_steps steps!"
            beta = 0.0
        end

        if cg_steps == cg_steps_max
            cg_steps_max > 1 && @info "Max. CG steps reached, resetting CG."
            beta = 0.0
        end

        if step < 1e-5
            cg_steps_max > 1 && @warn "Previous step was very small, resetting CG after $cg_steps steps!"
            beta = 0.0
        end

        step_dir = beta == 0 ? grad : grad + beta * step_dir #line search steps with -step_dir, hence grad, not -grad
        cg_steps = beta == 0 ? 1 : cg_steps + 1

        if beta == 0
            step_internal = step
        end
        
        En_prev = En
        step_corr = min(max(step_internal, 0.001),0.1)
        step_internal, En = line_search_energy(M, En, step_dir, norm_grad^2, step_corr, hMPO,
                                                        max_attempts=(beta==0 ? 3 : 1))
        if En > En_prev && beta != 0
            @warn "Line search increased the energy, resetting CG after $cg_steps steps!"
            step_internal = 0.0
            En = En_prev
            step_dir = grad
        end
        println("$k, $norm_grad, $step_internal, $En, $(En-En_prev)")
        if norm_grad < tol
            break
        end
        
        ts[k] = toq()
        betas[k] = beta
        ngs[k] = norm_grad
        ens[k] = En
        steps[k] = step_internal

        if step_internal != 0
            Anew = mps_tensor(M) .- real(T)(step_internal) .* step_dir
            set_mps_tensor!(M, Anew)
            normalize!(M)
        end
    end

    normalize!(M)
    M, norm_grad, (ts, ens, ngs, steps, betas)
end