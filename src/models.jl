function ising_local_MPO(::Type{T}, shift::Number=0.0; hz::Number=1.0, hx::Number=0.0)::MPO_open{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    h1_11 = -hz*Z - hx*X + shift*I

    h1 = zeros(eltype(h1_11), 2,2,1,2)
    h2 = zeros(eltype(h1_11), 2,2,2,1)
    
    h1[:,:,1,1] = h1_11
    h1[:,:,1,2] = -X
    
    h2[:,:,1,1] = E
    h2[:,:,2,1] = X
    
    h1 = permutedims(h1, (3,2,4,1)) #[m1,ket,m2,bra]
    h2 = permutedims(h2, (3,2,4,1)) #[m1,ket,m2,bra]
    
    h1 = convert(MPOTensor{T}, h1)
    h2 = convert(MPOTensor{T}, h2)
    
    MPOTensor{T}[h1, h2]
end

function ising_PBC_MPO(::Type{T}; hz::Real=1.0, hx::Real=0.0)::MPO_PBC_uniform{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,3,3)
    hB = zeros(Float64, 2,2,3,3)
    
    hM[:,:,1,1] = E
    hM[:,:,2,1] = X
    hM[:,:,3,1] = -hz*Z - hx*X
    hM[:,:,3,2] = -X
    hM[:,:,3,3] = E
    
    hB[:,:,1,:] = hM[:,:,3,:]
    hB[:,:,2,3] = X
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hB = permutedims(hB, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hB = convert(MPOTensor{T}, hB)
    
    (hB, hM)
end

function ising_OBC_MPO(::Type{T}; hz::Real=1.0, hx::Real=0.0)::MPO_open_uniform{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,3,3)
    
    hM[:,:,1,1] = E
    hM[:,:,2,1] = X
    hM[:,:,3,1] = -hz*Z - hx*X
    hM[:,:,3,2] = -X
    hM[:,:,3,3] = E
    
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

function ising_PBC_MPO_split(::Type{T}; hz::Real=1.0, hx::Real=0.0)::MPO_PBC_uniform_split{T} where {T}
    X = [0.0 1.0; 1.0 0.0]
    hL = reshape(-X, (2,2,1,1))
    hR = reshape(X, (2,2,1,1))
    hL = convert(MPOTensor{T}, permutedims(hL, (3,2,4,1)))
    hR = convert(MPOTensor{T}, permutedims(hR, (3,2,4,1)))
    
    h_B = MPOTensor{T}[hL, hR]
    
    (ising_OBC_MPO(T, hz=hz, hx=hx), h_B)
end

function ising_Hn_MPO_split(::Type{T}, n::Integer, N::Integer; hz::Real=1.0, hx::Real=0.0) where {T}
    (hL, hM, hR), (hb1, hb2) = ising_PBC_MPO_split(T, hz=hz, hx=hx)
    
    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,2,:] *= cis(n*1.5*2π/N)

    hR[3,:,1,:] *= cis(n*N*2π/N) #I realise this is not strictly necessary, but it shows intent! :)

    get_hM = (j::Integer)->begin
        hM_j = copy(hM)
        hM_j[3,:,1,:] *= cis(n*j*2π/N)
        hM_j[3,:,2,:] *= cis(n*(j+0.5)*2π/N)
        hM_j
    end

    Hn_OBC = MPOTensor{T}[hL, (get_hM(n) for n in 2:N-1)..., hR]

    hb1 *= cis(n*(N+0.5)*2π/N)
    
    Hn_b = MPOTensor{T}[hb1, hb2]

    n, (Hn_OBC, Hn_b)
end

#------------------------------
function heis_local_MPO(::Type{T}; J::Number=1, spin::Real=1//2)::MPO_open{T} where {T}
    if spin == 1//2
        E = Matrix{ComplexF64}(I,2,2)
        X = complex([0.0 1.0; 1.0 0.0])
        Y = 1.0im*[0.0 -1.0; 1.0 0.0]
        Z = complex([1.0 0.0; 0.0 -1.0])
        zer = zero(E)
    else
        ValueError()
    end

    h1 = J*[X Y Z]
    h2 = [X;
          Y;
          Z]

    h1 = reshape(h1, (2,1,2,3))
    h2 = reshape(h2, (2,3,2,1))

    h1 = permutedims(h1, (2,3,4,1))
    h2 = permutedims(h2, (2,3,4,1))

    MPOTensor{T}[convert(MPOTensor{T}, h1), convert(MPOTensor{T}, h2)]
end

#------------------------------

"""
-(lamda*X1*X2 + delta1*X1*X3 + delta2*Z1*Z2 + hz*Z1)
"""
function ANNNI_local_MPO(::Type{T}; hz::Real=1.0, delta1::Real=0.0, delta2::Real=0.0, lambda::Real=1.0)::MPO_open{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    h1 = zeros(Float64, 2,2,1,2)
    h2 = zeros(Float64, 2,2,2,2)
    h3 = zeros(Float64, 2,2,2,1)
    
    h1[:,:,1,1] = -X
    h1[:,:,1,2] = -Z
    
    h2[:,:,1,1] = lambda*X
    h2[:,:,1,2] = E
    h2[:,:,2,1] = delta2 * Z + I*hz
    
    h3[:,:,1,1] = E
    h3[:,:,2,1] = delta1 * X
    
    h1 = permutedims(h1, (3,2,4,1)) #[m1,ket,m2,bra]
    h2 = permutedims(h2, (3,2,4,1)) #[m1,ket,m2,bra]
    h3 = permutedims(h3, (3,2,4,1)) #[m1,ket,m2,bra]
    
    h1 = convert(MPOTensor{T}, h1)
    h2 = convert(MPOTensor{T}, h2)
    h3 = convert(MPOTensor{T}, h3)
    
    MPOTensor{T}[h1, h2, h3]
end

function ANNNI_OBC_MPO(::Type{T}; hz::Real=1.0, delta1::Real=0.0, delta2::Real=0.0, lambda::Real=1.0)::MPO_open_uniform{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    
    hM = zeros(Float64, 2,2,5,5)
    
    hM[:,:,1,1] = E
    hM[:,:,2,1] = X
    hM[:,:,3,1] = Z
    hM[:,:,4,2] = E
    hM[:,:,5,2] = -lambda*X
    hM[:,:,5,3] = -delta2*Z-hz*I
    hM[:,:,5,4] = -delta1*X
    hM[:,:,5,5] = E
    
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

function ANNNI_PBC_MPO_split(::Type{T}; hz::Real=1.0, delta1::Real=0.0, delta2::Real=0.0, lambda::Real=1.0)::MPO_PBC_uniform_split{T} where {T}
    #Turn off hz for boundary terms, since they're already in the OBC part.
    hL, hM, hR = ANNNI_OBC_MPO(T, hz=0.0, delta1=delta1, delta2=delta2, lambda=lambda)
    
    #Pick out only the needed terms from the OBC Hamiltonian. Reduces the max. bond dimension to 3.
    hL = hL[:,:,4:5,:]
    hM1 = hM[4:5,:,2:4,:]
    hM2 = hM[2:4,:,1:2,:]
    hR = hR[1:2,:,:,:]
    
    h_B = MPOTensor{T}[hL, hM1, hM2, hR]
    
    (ANNNI_OBC_MPO(T, hz=hz, delta1=delta1, delta2=delta2, lambda=lambda), h_B)
end

function ANNNI_Hn_MPO_split(::Type{T}, n::Integer, N::Integer; hz::Real=1.0, delta1::Real=0.0, delta2::Real=0.0, lambda::Real=1.0) where {T}
    (hL, hM, hR), (hb1, hb2, hb3, hb4) = ANNNI_PBC_MPO_split(T, hz=hz, delta1=delta1, delta2=delta2, lambda=lambda)
    
    Z = [1.0 0.0; 0.0 -1.0]

    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,2,:] *= cis(n*1.5*2π/N)
    hL[1,:,3,:] = -delta2 * cis(n*(1.5)*2π/N) * Z - hz * cis(n*2*2π/N) * I
    hL[1,:,4,:] *= cis(n*2*2π/N)

    get_hM = (j::Integer)->begin
        hM_j = copy(hM)
        hM_j[5,:,2,:] *= cis(n*(j+0.5)*2π/N)
        hM_j[5,:,3,:] = -delta2 * cis(n*(j+0.5)*2π/N) * Z - hz * cis(n*(j+1)*2π/N) * I
        hM_j[5,:,4,:] *= cis(n*(j+1)*2π/N)
        hM_j
    end

    Hn_OBC = MPOTensor{T}[hL, (get_hM(n) for n in 2:N-1)..., hR]

    #Boundary MPO tensor for site N-1
    hb1[1,:,1,:] *= cis(n*N*2π/N)

    #Boundary tensor for site N
    hb2[2,:,1,:] *= cis(n*0.5*2π/N)
    hb2[2,:,2,:] = -delta2 * cis(n*0.5*2π/N) * Z
    hb2[2,:,3,:] *= cis(n*1*2π/N)
    
    Hn_b = MPOTensor{T}[hb1, hb2, hb3, hb4]

    n, (Hn_OBC, Hn_b)
end

#------------------------------

function OBF_local_MPO(::Type{T}; lambda1::Real=1.0, lambda2::Real=1.0,
                            lambdaC::Real=0.0)::MPO_open{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    Y = [0.0im -1.0im; 1.0im 0.0im]
    
    Tin = (lambdaC == 0.0 ? Float64 : ComplexF64)
    d = (lambdaC == 0.0 ? 2 : 3)

    h1 = zeros(Tin, 2,2,1,3)
    h2 = zeros(Tin, 2,2,3,d)
    h3 = zeros(Tin, 2,2,d,1)
    
    h1[:,:,1,1] = lambda2*Z
    h1[:,:,1,2] = lambda2*X - lambda1*E
    h1[:,:,1,3] = E
    
    h2[:,:,1,1] = Z
    h2[:,:,2,2] = Z
    h2[:,:,3,1] = -lambda1*E
    
    h3[:,:,1,1] = X
    h3[:,:,2,1] = Z

    if lambdaC != 0.0
        h2[:,:,3,2] = lambdaC * Y
        h2[:,:,3,3] = -lambdaC * Z

        h3[:,:,3,1] = Y
    end
    
    h1 = permutedims(h1, (3,2,4,1)) #[m1,ket,m2,bra]
    h2 = permutedims(h2, (3,2,4,1)) #[m1,ket,m2,bra]
    h3 = permutedims(h3, (3,2,4,1)) #[m1,ket,m2,bra]
    
    h1 = convert(MPOTensor{T}, h1)
    h2 = convert(MPOTensor{T}, h2)
    h3 = convert(MPOTensor{T}, h3)
    
    MPOTensor{T}[h1, h2, h3]
end

function OBF_OBC_MPO(::Type{T}; lambda1::Real=1.0, lambda2::Real=1.0,
                        lambdaC::Real=0.0)::MPO_open_uniform{T} where {T}
    E = Matrix{Float64}(I,2,2)
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]
    Y = [0.0im -1.0im; 1.0im 0.0im]

    Tin = (lambdaC == 0.0 ? Float64 : ComplexF64)
    hM = zeros(Tin, 2,2,6,6)
    
    hM[:,:,1,1] = E
    hM[:,:,2,1] = X
    hM[:,:,3,1] = Z
    hM[:,:,4,2] = lambda2*Z #Z
    hM[:,:,5,3] = Z
    hM[:,:,6,1] = -lambda1*X
    hM[:,:,6,3] = -lambda1*Z
    hM[:,:,6,4] = Z #lambda2*Z
    hM[:,:,6,5] = lambda2*X
    hM[:,:,6,6] = E

    if lambdaC != 0.0
        hM[:,:,4,1] = -lambdaC*Y
        hM[:,:,6,3] += lambdaC*Y
    end
    
    hL = zeros(Tin, 2,2,1,6)
    hR = zeros(Tin, 2,2,6,1)
    
    hL[:,:,1,:] = hM[:,:,6,:]
    hR[:,:,:,1] = hM[:,:,:,1]
    
    hM = permutedims(hM, (3,2,4,1)) #[m1,ket,m2,bra]
    hL = permutedims(hL, (3,2,4,1)) #[m1,ket,m2,bra]
    hR = permutedims(hR, (3,2,4,1)) #[m1,ket,m2,bra]
    
    hM = convert(MPOTensor{T}, hM)
    hL = convert(MPOTensor{T}, hL)
    hR = convert(MPOTensor{T}, hR)
    
    (hL, hM, hR)
end

function OBF_PBC_MPO_split(::Type{T}; lambda1::Real=1.0, lambda2::Real=1.0,
                                lambdaC::Real=0.0)::MPO_PBC_uniform_split{T} where {T}
    hL, hM, hR = OBF_OBC_MPO(T, lambda1=lambda1, lambda2=lambda2, lambdaC=lambdaC)
    
    #Pick out only the needed terms from the OBC Hamiltonian. Reduces the max. bond dimension to 3.
    hL = hL[:,:,4:6,:]
    hM1 = hM[4:6,:,2:5,:]
    hM2 = hM[2:5,:,1:3,:]
    hR = hR[1:3,:,:,:]
    
    h_B = MPOTensor{T}[hL, hM1, hM2, hR]
    
    (OBF_OBC_MPO(T, lambda1=lambda1, lambda2=lambda2, lambdaC=lambdaC), h_B)
end


function OBF_Hn_MPO_split(::Type{T}, n::Integer, N::Integer; lambda1::Real=1.0, lambda2::Real=1.0, 
    ZZXoff::Number=5/4, XZZoff::Number=3/4) where {T}
    (hL, hM, hR), (hb1, hb2, hb3, hb4) = OBF_PBC_MPO_split(T, lambda1=lambda1, lambda2=lambda2)
    
    Z = [1.0 0.0; 0.0 -1.0]

    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,3,:] *= cis(n*1.5*2π/N)

    #hL[1,:,4,:] *= cis(n*(1 + ZZXoff)*2π/N) #ZZX
    hL[1,:,5,:] *= cis(n*(1 + XZZoff)*2π/N) #XZZ

    get_hM = (j::Integer)->begin
        hM_j = copy(hM)
        hM_j[6,:,1,:] *= cis(n*j*2π/N)
        hM_j[6,:,3,:] *= cis(n*(j+0.5)*2π/N)
        #hM_j[6,:,4,:] *= cis(n*(j+ZZXoff)*2π/N) #ZZX
        hM_j[4,:,2,:] *= cis(n*(j-1+ZZXoff)*2π/N) #ZZX
        hM_j[6,:,5,:] *= cis(n*(j+XZZoff)*2π/N) #XZZ
        hM_j
    end

    Hn_OBC = MPOTensor{T}[hL, (get_hM(n) for n in 2:N-1)..., hR]

    #Boundary tensor for site N-1
    #hb1[1,:,1,:] *= cis(n*(N-1 + ZZXoff)*2π/N) #ZXX
    hb1[1,:,2,:] *= cis(n*(N-1 + XZZoff)*2π/N) #XXZ (N-1)

    #Boundary tensor for site N
    hb2[3,:,2,:] *= cis(n*0.5*2π/N)
    #hb2[3,:,3,:] *= cis(n*ZZXoff*2π/N) #ZZX
    hb2[1,:,1,:] *= cis(n*(N-1 + ZZXoff)*2π/N) #ZZX (N-1)
    hb2[3,:,4,:] *= cis(n*XZZoff*2π/N) #XXZ (N)

    #Boundary tensor for site 1
    hb3[3,:,2,:] *= cis(n*(ZZXoff)*2π/N) #ZZX (N)
    
    Hn_b = MPOTensor{T}[hb1, hb2, hb3, hb4]

    n, (Hn_OBC, Hn_b)
end

#------------------------------

function weylops(p::Integer)
    om = cis(2π / p)
    U = diagm(ComplexF64[om^j for j in 0:p-1])
    V = diagm(ones(p - 1), 1)
    V[end, 1] = 1
    U, V, om
end

function potts3_OBC_MPO(::Type{T}; h::Real=1.0) where {T}
    E = Matrix{ComplexF64}(I,3,3)
    U, V, om = weylops(3)
    
    hM = zeros(ComplexF64, 3,3,4,4)

    hM[:,:,1,1] = E
    hM[:,:,2,1] = U
    hM[:,:,3,1] = U'
    hM[:,:,4,1] = -h/2 * (V+V')
    hM[:,:,4,2] = -0.5*U'
    hM[:,:,4,3] = -0.5*U
    hM[:,:,4,4] = E
    
    hL = zeros(ComplexF64, 3,3,1,4)
    hR = zeros(ComplexF64, 3,3,4,1)
    
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

function potts3_local_MPO(::Type{T}; h::Real=1.0) where {T}
    hL, hM, hR = potts3_OBC_MPO(T, h=h)
    
    MPOTensor{T}[hL[:,:,1:3,:], hR[1:3,:,:,:]]
end

function potts3_PBC_MPO_split(::Type{T}; h::Real=1.0)::MPO_PBC_uniform_split{T} where {T}
    hL, hM, hR = potts3_OBC_MPO(T, h=h)
    
    h_B = MPOTensor{T}[hL[:,:,2:3,:], hR[2:3,:,:,:]]
    
    ((hL, hM, hR), h_B)
end

function potts3_Hn_MPO_split(::Type{T}, n::Integer, N::Integer; h::Real=1.0) where {T}
    (hL, hM, hR), (hb1, hb2) = potts3_PBC_MPO_split(T, h=h)
    
    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,2,:] *= cis(n*1.5*2π/N)
    hL[1,:,3,:] *= cis(n*1.5*2π/N)

    hR[4,:,1,:] *= cis(n*N*2π/N)

    get_hM = (j::Integer)->begin
        hM_j = copy(hM)
        hM_j[4,:,1,:] *= cis(n*j*2π/N)
        hM_j[4,:,2,:] *= cis(n*(j+0.5)*2π/N)
        hM_j[4,:,3,:] *= cis(n*(j+0.5)*2π/N)
        hM_j
    end

    Hn_OBC = MPOTensor{T}[hL, (get_hM(n) for n in 2:N-1)..., hR]

    hb1 *= cis(n*0.5*2π/N)
    
    Hn_b = MPOTensor{T}[hb1, hb2]

    n, (Hn_OBC, Hn_b)
end