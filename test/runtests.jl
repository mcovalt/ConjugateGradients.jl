#!/usr/bin/env julia

#Start Test Script
using ConjugateGradients
using Base.Test

function test_cg()
    tA = sprandn(100,100,.1) + 10.0*speye(100)
    A = tA'*tA
    b = rand(100)
    true_x = A\b
    x, exit_code, num_iters = cg((x,y) -> A_mul_B!(x, A, y) , b)
    if norm(true_x - x) < 1e-6
        return true
    else
        return false
    end
end

function test_bicgstab()
    A = sprandn(100,100,.1) + 10.0*speye(100)
    b = rand(100)
    true_x = A\b
    x, exit_code, num_iters = bicgstab((x,y) -> A_mul_B!(x, A, y) , b)
    if norm(true_x - x) < 1e-6
        return true
    else
        return false
    end
end

@test test_cg()
@test test_bicgstab()