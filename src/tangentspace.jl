
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