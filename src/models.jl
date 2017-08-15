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

#------------------------------

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

function ANNNI_Hn_MPO_split{T}(::Type{T}, n::Int, N::Int; hz::Float64=1.0, delta1::Float64=0.0, delta2::Float64=0.0, lambda::Float64=1.0)
    (hL, hM, hR), (hb1, hb2, hb3, hb4) = ANNNI_PBC_MPO_split(T, hz=hz, delta1=delta1, delta2=delta2, lambda=lambda)
    
    Z = [1.0 0.0; 0.0 -1.0]

    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,2,:] *= cis(n*1.5*2π/N)
    hL[1,:,3,:] = -delta2 * cis(n*(1.5)*2π/N) * Z - hz * cis(n*2*2π/N) * I
    hL[1,:,4,:] *= cis(n*2*2π/N)

    get_hM = (j::Int)->begin
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

    n, Hn_OBC, Hn_b
end

#------------------------------

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

function potts3_Hn_MPO_split{T}(::Type{T}, n::Int, N::Int; h::Float64=1.0)
    (hL, hM, hR), (hb1, hb2) = potts3_PBC_MPO_split(T, h=h)
    
    hL[1,:,1,:] *= cis(n*2π/N)
    hL[1,:,2,:] *= cis(n*1.5*2π/N)
    hL[1,:,3,:] *= cis(n*1.5*2π/N)

    hR[4,:,1,:] *= cis(n*N*2π/N)

    get_hM = (j::Int)->begin
        hM_j = copy(hM)
        hM_j[4,:,1,:] *= cis(n*j*2π/N)
        hM_j[4,:,2,:] *= cis(n*(j+0.5)*2π/N)
        hM_j[4,:,3,:] *= cis(n*(j+0.5)*2π/N)
        hM_j
    end

    Hn_OBC = MPOTensor{T}[hL, (get_hM(n) for n in 2:N-1)..., hR]

    hb1 *= cis(n*0.5*2π/N)
    
    Hn_b = MPOTensor{T}[hb1, hb2]

    n, Hn_OBC, Hn_b
end