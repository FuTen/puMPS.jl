@testset "MPS Tensors and Transfer Matrices" begin
    d1 = 3; d2 = 2; AD1 = 4; AD2 = 5; BD1 = 2; BD2 = 3
    
    A1 = rand_MPSTensor(Float64,d1,AD1,AD2)
    @test bond_dim(A1) == AD1
    @test bond_dim_R(A1) == AD2
    @test phys_dim(A1) == d1

    A1 = rand_MPSTensor_unitary(Float64,d1,AD1,AD2)
    @test bond_dim(A1) == AD1
    @test bond_dim_R(A1) == AD2
    @test phys_dim(A1) == d1
    
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        U = puMPS.MPS.randunitary(T,3)
        @test U  * U' ≈ Matrix{T}(I,3,3)
        @test U' * U  ≈ Matrix{T}(I,3,3)

        A1 = rand_MPSTensor_unitary(T,d1,AD1,AD2)
        @test eltype(A1) == T

        TM1 = TM_convert(TM_dense(A1,A1))
        x = Matrix{T}(I,AD2,AD2)
        y = applyTM_r(A1,A1,x)
        @test eltype(y) == T
        @test all(x - I .≈ 0)
        @tensor y2[t,b] := TM1[t,b,ti,bi] * x[ti,bi]
        @test y2 ≈ y

        B1 = rand_MPSTensor(T,d1,BD1,BD2)
        @test eltype(B1) == T
        x = rand(T,AD1,BD1)
        y = applyTM_l(A1,B1,x)
        TM1 = TM_convert(TM_dense(A1,B1))
        @test eltype(TM1) == T
        @tensor y2[t,b] := conj(TM1[ti,bi,t,b]) * x[ti,bi]
        @test y2 ≈ y

        A2 = rand_MPSTensor(T,d2,AD2,AD2)
        B2 = rand_MPSTensor_unitary(T,d2,BD2,BD2)
        TM2 = TM_convert(TM_dense(A2,B2))
        @tensor TM12[t1,b1,t2,b2] := TM1[t1,b1,ti,bi] * TM2[ti,bi,t2,b2]

        TM1 = TM_convert(TM1)
        TM2 = TM_convert(TM2)
        TM12 = TM_convert(TM12)
        
        TM12_2 = applyTM_r(A1,B1,TM2)
        @test eltype(TM12_2) == T
        @test TM12_2 ≈ TM12
        
        TM12_2 = applyTM_l(A2,B2,TM1)
        @test eltype(TM12_2) == T
        @test TM12_2 ≈ TM12

        wrk = workvec_applyTM_r(A1,B1,TM2)
        fill!(TM12_2, 0.0)
        res = applyTM_r!(TM12_2,A1,B1,TM2,wrk)
        @test res === TM12_2
        @test TM12_2 ≈ TM12

        wrk = workvec_applyTM_l(A2,B2,TM1)
        fill!(TM12_2, 0.0)
        res = applyTM_l!(TM12_2,A2,B2,TM1,wrk)
        @test res === TM12_2
        @test TM12_2 ≈ TM12
    end
end

@testset "MPS TM eigensolvers" begin
    d = 3; AD = 4; BD = 2
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = rand_MPSTensor_unitary(T,d,AD,AD)
        B = rand_MPSTensor_unitary(T,d,BD,BD)
        ev, evmL, evmR = tm_eigs_dense(A,B)
        srt = sortperm(ev, by=abs, rev=true)
        ev = ev[srt]
        evmL = evmL[srt]
        evmR = evmR[srt]

        for j = 1:length(ev)
            y = applyTM_r(A,B,evmR[j])
            @test y ≈ evmR[j] * ev[j]

            y = applyTM_l(A,B,evmL[j])
            @test y ≈ evmL[j] * conj(ev[j])
        end

        evlS, evrS, evmLS, evmRS = tm_eigs(A,B,4,D_dense_max=0)
        @test argmax(abs.(evlS)) == argmax(abs.(evrS)) == 1
        @show T
        @test evlS[1] ≈ ev[1]
        @test evrS[1] ≈ ev[1]
    end
end

@testset "MPS/MPO Tensors and Transfer Matrices" begin
    d1 = 3; d2 = 2; AD1 = 4; AD2 = 5; BD1 = 2; BD2 = 3
    M1 = 2; M2 = 4
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        O1 = rand_MPOTensor(T,d1,M1,M2)
        A1 = rand_MPSTensor_unitary(T,d1,AD1,AD2)
        B1 = rand_MPSTensor_unitary(T,d1,BD1,BD2)
        TM1 = TM_dense_MPO(A1,B1,O1)

        O2 = rand_MPOTensor(T,d2,M2,M1)
        A2 = rand_MPSTensor(T,d2,AD2,AD1)
        B2 = rand_MPSTensor(T,d2,BD2,BD1)
        TM2 = TM_dense_MPO(A2,B2,O2)

        @tensor TM12[t1,m1,b1,t2,m2,b2] := TM1[t1,m1,b1,ti,mi,bi] * TM2[ti,mi,bi,t2,m2,b2]

        TM12_2 = applyTM_MPO_r(A1,B1,O1,TM2)
        @test TM12_2 ≈ TM12
        TM12_2 = applyTM_MPO_l(A2,B2,O2,TM1)
        @test TM12_2 ≈ TM12

        wrk = workvec_applyTM_MPO_r(A1,B1,O1,TM2)
        fill!(TM12_2, 0.0)
        res = applyTM_MPO_r!(TM12_2,A1,B1,O1,TM2,wrk)
        @test res === TM12_2
        @test TM12_2 ≈ TM12

        wrk = workvec_applyTM_MPO_l(A2,B2,O2,TM1)
        fill!(TM12_2, 0.0)
        res = applyTM_MPO_l!(TM12_2,A2,B2,O2,TM1,wrk)
        @test res === TM12_2
        @test TM12_2 ≈ TM12
    end
end