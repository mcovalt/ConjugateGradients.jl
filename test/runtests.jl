using ConjugateGradients
using LinearAlgebra
using SparseArrays
using Test

function test_cg(n=100)
    tA = sprandn(n, n, 0.1) + spdiagm(0=>fill(10.0, n))
    A = tA'*tA
    b = rand(n)
    true_x = A\b
    x, exit_code, num_iters = cg((x,y) -> mul!(x, A, y), b)
    norm(true_x - x) < 1e-6
end

function test_bicgstab(n=100)
    A = sprandn(n, n, 0.1) + spdiagm(0=>fill(10.0, n))
    b = rand(n)
    true_x = A\b
    x, exit_code, num_iters = bicgstab((x,y) -> mul!(x, A, y), b)
    norm(true_x - x) < 1e-6
end

@testset "ConjugateGradients" begin
    @test test_cg()
    @test test_bicgstab()
end
