
"""
    Represents a tangent vector, with momentum `p=2π/N*k`, living in the tangent space of a puMPState `state`.
"""
type puMPSTvec{T}
    state::puMPState{T}
    B::MPSTensor{T}
    k::Float64 #p=2π/N*k
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
    
    TAA = length(TAA_all) > 0 ? TAA_all[1] : (length(O) > 0 ? TM_dense_MPO(A1, A2, O[1]) : TM_convert(TM_dense(A1, A2)))
    
    TBBs = length(O) > 0 ? TM_dense_MPO(B1, B2, O[1]) : TM_convert(TM_dense(B1, B2))
    scale!(TBBs, cis(p1-p2))
    
    TABs = length(O) > 0 ? TM_dense_MPO(A1, B2, O[1]) : TM_convert(TM_dense(A1, B2))
    scale!(TABs, cis(-p2))
    
    if length(TBAs_all) > 0
        TBAs = TBAs_all[1]
    else
        TBAs = length(O) > 0 ? TM_dense_MPO(B1, A2, O[1]) : TM_convert(TM_dense(B1, A2))
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
        for n in max(2, length(O)+1):N
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

function fidelity{T}(Tvec2::puMPSTvec{T}, Tvec1::puMPSTvec{T})
    N = num_sites(Tvec1)
    @assert num_sites(Tvec2) == N
    overlap(Tvec2, MPOTensor{T}[], Tvec1)
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

function tspace_ops_scale!{T}(ops::Vector{Array{T,6}}, N::Int)
    for op in ops
        scale!(op, N)
    end
    ops
end

"""
tangent_space_metric{T}(M::puMPState{T}, ks::Vector{<:Real}, blkTMs::Vector{MPS_TM{T}})

Computes the physical metric induced on the tangent space of the puMPState `M` for the tangent-space
momentum sectors specified in `ks`. The momenta are `ps = 2π/N .* ks` where `N = num_sites(M)`. 
"""
function tangent_space_metric{T}(M::puMPState{T}, ks::Vector{<:Real}, blkTMs::Vector{MPS_TM{T}})
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
    
    gc_enable(false)
    Gpart = ncon((A,conj(A), blkTMs[N-2]), ([2,-2,-4],[-3,-5,1],[-6,1,2,-1]))
    gc_enable(true)
    gc(false)
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]), Gpart, Gs[j])
    end
    
    for i in 2:N-2
        gc_enable(false)
        left_T = ncon((A, blkTMs[i-1]), ([-1,-2,1],[1,-3,-4,-5])) #gap on the top-left, then usual TM
        gc_enable(true)
        gc(false)
        gc_enable(false)
        right_T = ncon((conj(A), blkTMs[N-i-1]), ([-3,-2,1],[-1,1,-4,-5])) #gap on the bottom-right
        gc_enable(true)
        gc(false)
        
        gc_enable(false)
        Gpart = ncon((left_T, right_T), ([2,-2,-3,-4,1],[-6,-5,1,2,-1])) #complete loop, cost O(d^2 * D^6)
        gc_enable(true)
        gc(false)
        for j in 1:length(ps)
            BLAS.axpy!(cis(ps[j]*i), Gpart, Gs[j])
        end
    end
    
    Gpart = ncon((A, blkTMs[N-2], conj(A)), ([-6,-2,2],[2,-3,-4,1],[1,-5,-1]))
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]*(N-1)), Gpart, Gs[j])
    end
    
    tspace_ops_scale!(Gs, N)

    Gs
end

function tangent_space_MPO{T}(M::puMPState{T}, op::MPO_PBC_uniform{T}, ks::Vector{<:Real})
    #Let's do this like in tangent_space_metric(), but with MPO transfer matrices.
    A = mps_tensor(M)
    D = bond_dim(M)
    d = phys_dim(M)
    N = num_sites(M)
    ps = 2π/N .* ks
    
    blkTMs_op = blockTMs_MPO(M, op) #requires N * D^4 * M^2 bytes, with M the MPO bond dimension
    
    op_boundary, op_bulk = op
    #NOTE: blkTMs_op have HM right up until the full TM.. blkTMs[N] ends with HB. Each term below must add an HB somewhere.
    
    op_effs = Array{T,6}[zeros(T, (D,d,D, D,d,D)) for j in 1:length(ks)]
    
    gc_enable(false)
    part = ncon((op_boundary, blkTMs_op[N-1]),([1,-5,2,-2],[-6,2,-3,-4,1,-1])) #This is the same-site term.
    gc_enable(true)
    gc(false)
    for j in 1:length(ps)
        BLAS.axpy!(1.0, part, op_effs[j])
    end
    
    #This is one of the terms in which the gaps are nearest-neighbours
    blk = blkTMs_op[N-2]
    gc_enable(false)
    @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = op_boundary[m3,Pt,m1,pb] * (conj(A[V2b,pb,b]) * 
                                                ((blk[V2t,m1,b, t,m2,V1b] * A[t,pt,V1t]) * op_bulk[m2,pt,m3,Pb]))
    gc_enable(true)
    gc(false)
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]), part, op_effs[j])
    end
        
    for i in 2:N-2 #i is the number of sites between conj(gap) and gap.
        #Block TM times A and an MPO tensor (we choose to put the boundary tensor HB here)
        blk = blkTMs_op[i-1]
        gc_enable(false)
        @tensor left_T[tl,ml,pl,bl, tr,mr,br] := op_boundary[ml,s,mi,pl] * (A[tl,s,ti] * blk[ti,mi,bl, tr,mr,br])
        gc_enable(true)
        gc(false)
        
        #Block TM times conj(A) and an MPO tensor
        blk = blkTMs_op[N-i-1]
        gc_enable(false)
        @tensor right_T[tl,pl,ml,bl, tr,mr,br] := op_bulk[ml,pl,mi,t] * (conj(A[bl,t,bi]) * blk[tl,mi,bi, tr,mr,br])
        gc_enable(true)
        gc(false)
        
        #Combine to form an Heff contribution at cost O(D^6)
        gc_enable(false)
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = (left_T[t,m1,Pb,V2b, V1t,m2,b] * right_T[V2t,Pt,m2,b, t,m1,V1b])
        gc_enable(true)
        gc(false)
        for j in 1:length(ps)
            BLAS.axpy!(cis(ps[j]*i), part, op_effs[j])
        end
    end
    
    #This is the other term in which the gaps are nearest-neighbours
    blk = blkTMs_op[N-2]
    @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = op_boundary[m3,pt,m1,Pb] * (A[V2t,pt,t] * 
                                                ((blk[t,m1,V2b, V1t,m2,b] * conj(A[b,pb,V1b])) * op_bulk[m2,Pt,m3,pb]))
    for j in 1:length(ps)
        BLAS.axpy!(cis(ps[j]*(N-1)), part, op_effs[j])
    end

    tspace_ops_scale!(op_effs, N)

    op_effs
end

function tangent_space_MPO_OBC!{T}(op_effs::Vector{Array{T,6}}, M::puMPState{T}, op::MPO_open_uniform{T}, ks::Vector{<:Real})
    N = num_sites(M)
    opL, opM, opR = op

    op_full = MPOTensor{T}[opL, (opM for j in 1:N-2)..., opR]

    tangent_space_MPO_OBC!(op_effs, M, op_full, ks)
end

"""
tangent_space_MPO_OBC!{T}(op_effs::Vector{Array{T,6}}, M::puMPState{T}, op::MPO_open{T}, ks::Vector{<:Real})

Compute the OBC part of the effective operator in the momentum sectors specified in `ks`. 
See `tangent_space_metric_and_MPO()`.
"""
function tangent_space_MPO_OBC!{T}(op_effs::Vector{Array{T,6}}, M::puMPState{T}, op::MPO_open{T}, ks::Vector{<:Real})
    A = mps_tensor(M)
    N = num_sites(M)
    ps = 2π/N .* ks

    op_1_s = squeeze(op[1],1) #[Pt,M,Pb]
    op_N_s = squeeze(op[N],3) #[M,Pt,Pb]
    
    #First construct block transfer matrices. A conjugate gap at site 1, where the Hamiltonian begins.
    #This is essentially an MPO TM, but with a physical index on the left instead of an MPO index.
    op_n = op[2]
    gc_enable(false)
    @tensor blk_l[V1t,P,V1b, V2t,M,V2b] := ((((A[V1t,p1,vt] * op_1_s[p1,m,P]) 
                                             * A[vt,p2t,V2t]) * op_n[m,p2t,M,p2b]) * conj(A[V1b,p2b,V2b]))
    gc_enable(true)
    gc(false)

    #We also need block TM's for the latter part of the Hamiltonian. We will add the gap on the left later.
    #We precompute these right blocks, since we will use the intermediates
    blks_r = MPS_MPO_TM{T}[TM_dense_MPO(A, A, op[N])] #site N
    work = Vector{T}()
    for n in N-1:-1:2
        work = workvec_applyTM_MPO_r!(work, A, A, op[n], blks_r[end])
        res = res_applyTM_MPO_r(A, A, op[n], blks_r[end])
        push!(blks_r, applyTM_MPO_r!(res, A, A, op[n], blks_r[end], work))
    end
    
    #Same-site term
    blk = pop!(blks_r) #block for N-1 sites
    op_n = op[1]
    gc_enable(false)
    @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] := op_n[m2,Pt,m1,Pb] * blk[V2t,m1,V2b, V1t,m2,V1b]
    gc_enable(true)
    gc(false)
    for j in 1:length(ks)
        BLAS.axpy!(1.0, part, op_effs[j])
    end
    
    #Nearest-neighbour term with conjugate gap on the left, gap on the right. Conjugate gap is site 1.
    blk = pop!(blks_r) #block for N-2 sites
    op_1 = op[1]; op_n = op[2]
    gc_enable(false)
    @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = (op_1[m3,pt,m1,Pb] * (A[vt,pt,V1t] * 
                                                (op_n[m1,Pt,m2,pb] * (conj(A[V2b,pb,vb]) * blk[V2t,m2,vb, vt,m3,V1b]))))
    gc_enable(true)
    gc(false)
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]), part, op_effs[j])
    end
    
    #Terms separated by one or more sites
    for n in 3:N-1 #if conj(gap) is at site 1, gap is at site n
        blk_r = squeeze(pop!(blks_r), 5) #N-n sites
        op_n = op[n]
        gc_enable(false)
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = blk_l[vt,Pb,V2b, V1t,m1,vb1] * 
                                                  (op_n[m1,Pt,m2,pb] * (conj(A[vb1,pb,vb2]) * blk_r[V2t,m2,vb2, vt,V1b]))
        gc_enable(true)
        gc(false)
        for j in 1:length(ks)
            BLAS.axpy!(cis(ps[j]*(n-1)), part, op_effs[j])
        end
        
        work = workvec_applyTM_MPO_l!(work, A, A, op[n], blk_l)
        applyTM_MPO_l!(blk_l, A, A, op[n], blk_l, work) #extend blk_l by multiplying from the right by an MPO TM
    end
    
    #Nearest-neighbour term with gap on the left, conjugate gap on the right. Conjugate gap is site 1.
    gc_enable(false)
    @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = (blk_l[V2t,Pb,V2b, V1t,m,vb] * conj(A[vb,p,V1b])) * op_N_s[m,Pt,p]
    gc_enable(true)
    gc(false)
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]*(N-1)), part, op_effs[j])
    end

    op_effs
end

#This will return the transfer matrix (TM) from site nL to site nR, given a boundary Hamiltonian centred at the
#boundary between sites N and 1.
function blockTM_MPO_boundary{T}(M::puMPState{T}, op_b::MPO_open{T}, blkTMs::Vector{MPS_TM{T}}, nL::Int, nR::Int)
    A = mps_tensor(M)
    N = num_sites(M)
    Nop = length(op_b)
    
    @assert N > Nop
    @assert nL <= nR
    
    #Split op_b into left (starting at site 1) and right (ending at site N) parts
    op_b_L = op_b[Nop÷2+1:end] 
    op_b_R = op_b[1:Nop÷2]

    #Illustration for four-site boundary Hamiltonian:
    #
    # <-1-2-3-4-.-.-.-.-.-.-.-.-.-.-.-.-N->    state ket
    #   | | | |                     | | |
    # <-o-o | |        ...          | o-o->    operator
    #   | | | |                     | | |
    # <-1-2-3-4-.-.-.-.-.-.-.-.-.-.-.-.-N->    state bra
    #
    # Just the operator:
    #   | |                           | |
    # <-o-o                           o-o->
    #   | |                           | |
    # op_b_L                          op_b_R

    NopL = length(op_b_L)
    NopR = length(op_b_R)
    
    nL_in_op = nL - (N - NopR)
    nR_in_op = nR - (N - NopR)
    
    Nmid = min(nR, N-NopR) - max(nL, NopL + 1) + 1 #length of middle TM (no h_b terms)
    
    if Nmid > 0
        blk = TM_convert(blkTMs[Nmid])

        #These will add MPO terms if needed:
        #Note: The index ranges may be empty, in which case applyTM_MPO_x() returns blk unmodified.
        blk = applyTM_MPO_r(M, op_b_L[nL:end], blk)
        blk = applyTM_MPO_l(M, op_b_R[1:nR_in_op], blk)
    elseif nR <= length(op_b_L)
        blk = TM_dense_MPO(A, A, op_b_L[nR])
        blk = applyTM_MPO_r(M, op_b_L[nL:nR-1], blk)
    else
        blk = TM_dense_MPO(A, A, op_b_R[nL_in_op])
        blk = applyTM_MPO_l(M, op_b_R[nL_in_op+1:nR_in_op], blk)
    end
    
    blk
end

#This will return the boundary MPO tensor for site n. Where the MPO does not act on site n,
#return a zero-length result.
function op_term_MPO_boundary{T}(M::puMPState{T}, op_b::MPO_open{T}, n::Int)
    A = mps_tensor(M)
    N = num_sites(M)
    Nop = length(op_b)
    
    @assert N > Nop
    
    op_b_L = op_b[Nop÷2+1:end] 
    op_b_R = op_b[1:Nop÷2]
    
    NopL = length(op_b_L)
    NopR = length(op_b_R)
    
    if n <= length(op_b_L)
        return op_b_L[n]
    elseif n >= N-NopR+1
        return op_b_R[n-(N-NopR)]
    else
        return MPOTensor{T}((0,0,0,0)) #length zero MPO tensor
    end
end

"""
    tangent_space_MPO_boundary!{T}(op_effs::Vector{Array{T,6}}, M::puMPState{T}, op_b::MPO_open{T}, ks::Vector{<:Real}, blkTMs::Vector{MPS_TM{T}})

Compute the boundary part of the effective MPO in the momentum sectors specified in `ks`. 
See `tangent_space_metric_and_MPO()`.

This is very similar to the OBC MPO case, except that we now have a 
short MPO `op_b` centred on the boundary between sites N and 1.
"""
function tangent_space_MPO_boundary!{T}(op_effs::Vector{Array{T,6}}, M::puMPState{T}, op_b::MPO_open{T}, 
                                        ks::Vector{<:Real}, blkTMs::Vector{MPS_TM{T}})
    A = mps_tensor(M)
    N = num_sites(M)
    ps = 2π/N .* ks
    
    Nh = length(op_b)
    @assert iseven(Nh)
    
    #Check this is really OBC
    @assert size(op_b[1], 1) == 1
    @assert size(op_b[end], 3) == 1
    
    #On-site term
    blk = blockTM_MPO_boundary(M, op_b, blkTMs, 2, N)
    op_b_1 = op_term_MPO_boundary(M, op_b, 1)
    gc_enable(false)
    @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] := blk[V2t,m2,V2b, V1t,m1,V1b] * op_b_1[m1,Pt,m2,Pb]
    gc_enable(true)
    gc(false)
    for j in 1:length(ks)
        BLAS.axpy!(1.0, part, op_effs[j])
    end
    
    #NN-term 1, with conjugate gap (site 1) then gap
    blk = blockTM_MPO_boundary(M, op_b, blkTMs, 3, N)
    op_b_n = op_term_MPO_boundary(M, op_b, 2)
    gc_enable(false)
    if length(op_b_n) == 0
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = op_b_1[m2,pt,m1,Pb] * (A[vt,pt,V1t] * (conj(A[V2b,Pt,vb]) 
                * blk[V2t,m1,vb, vt,m2,V1b]))
    else
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = op_b_1[m3,pt,m1,Pb] * 
        (A[vt,pt,V1t] * (op_b_n[m1,Pt,m2,pb] * (conj(A[V2b,pb,vb]) * blk[V2t,m2,vb, vt,m3,V1b])))
    end
    gc_enable(true)
    gc(false)
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]), part, op_effs[j])
    end
    
    #gap at site n
    for n in 3:N-1
        blkL = blockTM_MPO_boundary(M, op_b, blkTMs, 2, n-1)
        gc_enable(false)
        @tensor blkL_cgap[V1t,M1,P,V1b, V2t,M2,V2b] := op_b_1[M1,p,m,P] * (A[V1t,p,vt] * blkL[vt,m,V1b, V2t,M2,V2b])
        gc_enable(true)
        gc(false)
        
        blkR = blockTM_MPO_boundary(M, op_b, blkTMs, n+1, N)
        op_b_n = op_term_MPO_boundary(M, op_b, n)
        gc_enable(false)
        if length(op_b_n) == 0
            @tensor blkR_gap[V1t,P,M1,V1b, V2t,M2,V2b] := conj(A[V1b,P,vb]) * blkR[V1t,M1,vb, V2t,M2,V2b]
        else
            #Note: This is never called for a nearest-neighbour boundary Hamiltonian
            @tensor blkR_gap[V1t,P,M1,V1b, V2t,M2,V2b] := op_b_n[M1,P,m,p] * (conj(A[V1b,p,vb]) * blkR[V1t,m,vb, V2t,M2,V2b])
        end
        gc_enable(true)
        gc(false)
        
        gc_enable(false)
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = blkL_cgap[vt,m2,Pb,V2b, V1t,m1,vb] * blkR_gap[V2t,Pt,m1,vb, vt,m2,V1b]
        gc_enable(true)
        gc(false)
        for j in 1:length(ks)
            BLAS.axpy!(cis(ps[j]*(n-1)), part, op_effs[j])
        end
    end
    
    #NN-term 2, with gap (site N) then conjugate gap (site 1)
    blk = blockTM_MPO_boundary(M, op_b, blkTMs, 2, N-1)
    op_b_n = op_term_MPO_boundary(M, op_b, N)
    gc_enable(false)
    if length(op_b_n) == 0
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = ((op_b_1[m2,pt,m1,Pb] * (A[V2t,pt,vt] * blk[vt,m1,V2b, V1t,m2,vb])) *
            conj(A[vb,Pt,V1b]))
    else
        @tensor part[V1b,Pb,V2b, V1t,Pt,V2t] = ((op_b_1[m3,pt,m1,Pb] * (A[V2t,pt,vt] * blk[vt,m1,V2b, V1t,m2,vb])) *
            conj(A[vb,pb,V1b])) * op_b_n[m2,Pt,m3,pb]
    end
    gc_enable(true)
    gc(false)
    for j in 1:length(ks)
        BLAS.axpy!(cis(ps[j]*(N-1)), part, op_effs[j])
    end

    op_effs
end

"""
    tangent_space_metric_and_MPO{T}(M::puMPState{T}, op::MPO_PBC_uniform{T}, ks::Vector{<:Real}, lambda_i::Matrix{T})

Computes the tangent-space metric and effective operator given a physical-space operator as a periodic, uniform MPO,
for the momentum sectors specified in `ks`. The momenta are `ps = 2π/N .* ks` where `N = num_sites(M)`. 
"""
function tangent_space_metric_and_MPO{T}(M::puMPState{T}, op::MPO_PBC_uniform{T}, ks::Vector{<:Real}, lambda_i::Matrix{T})
    @time Gs = tangent_space_metric(M, ks, blockTMs(M))
    @time op_effs = tangent_space_MPO(M, op, ks)

    Gs, op_effs
end

"""
    tangent_space_metric_and_MPO{T}(M::puMPState{T}, op::Union{MPO_PBC_uniform_split{T}, MPO_PBC_split{T}}, 
                                                 ks::Vector{<:Real}, lambda_i::Matrix{T})

Computes the tangent-space metric and effective operator given a physical-space operator as a combination
of an open MPO, representing for example the Hamiltonian with open boundary conditions (OBC), and a boundary MPO.
The sum of the OBC and boundary parts of the operator must be translation invariant (up to a phase), which we exploit
to reduce the cost of the computation.
Metrics and effective Hamiltonians are computed for the momentum sectors specified in `ks`. 
The momenta are `ps = 2π/N .* ks` where `N = num_sites(M)`.
"""
function tangent_space_metric_and_MPO{T}(M::puMPState{T}, op::Union{MPO_PBC_uniform_split{T}, MPO_PBC_split{T}}, 
                                                      ks::Vector{<:Real}, lambda_i::Matrix{T})
    A = mps_tensor(M)
    D = bond_dim(M)
    d = phys_dim(M)
    N = num_sites(M)
    
    bTMs = blockTMs(M) #requires N * D^4 bytes
    @time Gs = tangent_space_metric(M, ks, bTMs)
    
    op_bulk, op_boundary = op
    
    op_effs = Array{T,6}[zeros(T, (D,d,D, D,d,D)) for j in 1:length(ks)]
    @time tangent_space_MPO_boundary!(op_effs, M, op_boundary, ks, bTMs)
    blkTMs = nothing #free up this space
    @time tangent_space_MPO_OBC!(op_effs, M, op_bulk, ks)
    tspace_ops_scale!(op_effs, N)
    
    Gs, op_effs
end

"""
    tangent_space_MPO{T}(M::puMPState{T}, op::Union{MPO_PBC_uniform_split{T}, MPO_PBC_split{T}}, ks::Vector{<:Real})

Given an MPO representation of an operator `op`, split into a large OBC MPO and a boundary
MPO, this computes its representation on the tangent space of the puMPState `M` for the tangent
vector momenta (of the ket tangent vector) specified in `ks`.
The sum of the OBC and boundary parts of the operator must be translation invariant (up to a phase), which we exploit
to reduce the cost of the computation.
"""
function tangent_space_MPO{T}(M::puMPState{T}, op::Union{MPO_PBC_uniform_split{T}, MPO_PBC_split{T}}, ks::Vector{<:Real})
    A = mps_tensor(M)
    D = bond_dim(M)
    d = phys_dim(M)
    N = num_sites(M)

    op_bulk, op_boundary = op

    op_effs = Array{T,6}[zeros(T, (D,d,D, D,d,D)) for j in 1:length(ks)]

    @time tangent_space_MPO_boundary!(op_effs, M, op_boundary, ks, blockTMs(M))
    blkTMs = nothing #free up this space
    @time tangent_space_MPO_OBC!(op_effs, M, op_bulk, ks)
    tspace_ops_scale!(op_effs, num_sites(M))

    op_effs
end

function tspace_ops_to_center_gauge!{T}(ops::Vector{Array{T,6}}, lambda_i::Matrix{T})
    for op in ops
        gc_enable(false)
        @tensor op[V1b,Pb,V2b, V1t,Pt,V2t] = lambda_i[vb,V2b] * (op[V1b,Pb,vb, V1t,Pt,vt] * lambda_i[vt,V2t])
        gc_enable(true)
        gc(false)
    end
    ops
end

"""
    excitations!{T}(M::puMPState{T}, H::Union{MPO_PBC_uniform{T}, MPO_PBC_uniform_split{T}}, ks::Vector{Int}, num_states::Vector{Int}; pinv_tol::Real=1e-10)

Computes eigenstates of the effective Hamiltonian obtained by projecting `H` onto the tangent space of the puMPState `M`.
This is done in the momentum sectors specified in `ks`, where each entry of `k = ks[j]` specified a momentum `k*2pi/N`,
where `N` is the number of sites. 

The number of eigenstates to be computed for each momentum sector is specified in `num_states`.

The function returns a list of energies, a list of momenta (entries of `ks`), and a list of normalized tangent vectors.
"""
function excitations!{T}(M::puMPState{T}, H::Union{MPO_PBC_uniform{T}, MPO_PBC_uniform_split{T}}, ks::Vector{<:Real}, num_states::Vector{Int}; pinv_tol::Real=1e-10)
    M, lambda, lambda_i = canonicalize_left!(M)
    lambda_i = full(lambda_i)
    
    @time Gs, Heffs = tangent_space_metric_and_MPO(M, H, ks, lambda_i)
    tspace_ops_to_center_gauge!(Gs, lambda_i)
    tspace_ops_to_center_gauge!(Heffs, lambda_i)

    excitations(M, Gs, Heffs, lambda_i, ks, num_states, pinv_tol=pinv_tol)
end

function excitations{T,Tk<:Real}(M::puMPState{T}, Gs::Vector{Array{T,6}}, Heffs::Vector{Array{T,6}}, lambda_i::Matrix{T}, 
                        ks::Vector{Tk}, num_states::Vector{Int}; pinv_tol::Real=1e-12)
    D = bond_dim(M)
    d = phys_dim(M)
    Bshp = mps_tensor_shape(d, D)
    par_dim = prod(Bshp)
    
    ens = Vector{T}[]
    exs = Vector{puMPSTvec{T}}[]
    ks_rep = Vector{Tk}[]
    for j in 1:length(ks)
        println("k=$(ks[j])")
        G = reshape(Gs[j], (par_dim, par_dim))
        Heff = reshape(Heffs[j], (par_dim, par_dim))
        
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
            Bl = Bc_mat * lambda_i #We assume G and Heff were in the center gauge.
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
    Hn_in_basis{T}(M::puMPState{T}, Hn_split::Tuple{Int, MPO_PBC_split}, Tvec_basis::Vector{puMPSTvec{T}}, ks::Vector{<:Real})

Given an MPO representation of a Hamiltonian Fourier mode, split into a large OBC MPO and a boundary
MPO, `Hn_split`, computes its matrix elements in the basis of puMPState tangent vectors `Tvec_basis`
which are assumed to live in the tangent space of the puMPState `M`.
"""
function Hn_in_basis{T}(M::puMPState{T}, Hn_split::Tuple{Int, MPO_PBC_split{T}}, Tvec_basis::Vector{puMPSTvec{T}}, ks::Vector{<:Real})
    #Hn is a Fourier mode of the Hamiltonian density with "momentum" n * 2pi/N
    #Each entry in ks, times 2pi/N is the momentum of one of the excitations (the ket). 
    #Due to the way tangent_space_MPO() works, the momentum of the other excitation is automatically (ks[j] + n) * 2pi/N.
    ks = 1.0 * ks

    n = Hn_split[1]
    Hn_effs = tangent_space_MPO(M, Hn_split[2], ks)
    Ntvs = length(Tvec_basis)
    Tvspins = map(spin, Tvec_basis)

    Hn_in_basis = zeros(T, (Ntvs,Ntvs))
    for j in 1:Ntvs
        ind = findfirst(ks, Tvspins[j])
        if ind > 0
            Bj = tvec_tensor(Tvec_basis[j])
            Hn_eff = reshape(Hn_effs[ind], (length(Bj), length(Bj)))
            for k in 1:Ntvs
                if Tvspins[k] == Tvspins[j] + n
                    Bk = tvec_tensor(Tvec_basis[k])
                    Hn_in_basis[k,j] = dot(vec(Bk), Hn_eff * vec(Bj))
                end
            end
        end
    end
    
    Hn_in_basis
end