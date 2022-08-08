# ConjugateGradients.jl

`ConjugateGradients.jl` is a flexible, non-allocating Julia implementation of the conjugate gradient and biconjugate gradient stabilized methods.

## Requirements

* Julia 1.2 and up

## Installation

```julia
julia> ]
pkg> add ConjugateGradients
```

## Why use ConjugateGradients.jl?

There are a few great iterative solver packages available for Julia: [IterativeSolvers.jl](https://github.com/JuliaMath/IterativeSolvers.jl), [KrylovMethods.jl](https://github.com/lruthotto/KrylovMethods.jl), and [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl). These are all very well rounded and complete packages.

This package, `ConjugateGradients.jl`, is built around reducing allocations as much as possible for a particular type of problem. As far as I know, if your program will be using an iterative solver *within* another iterative process, this module will result in less allocations compared to the previously mentioned packages<sup>*</sup>.

Also, in other iterative solvers, calls to BLAS functions are preferred for obvious reasons. This package uses Julia's multiple dispatch functionality to decide whether to use BLAS or native Julia code to make calculations based on the type associated with the arrays. This gives greater flexibility with types not represented by floating point numbers.

_<sup>*</sup> Hint: take a look at [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl) if this type of solver would be beneficial to your project. Combined, these packages can help reduce allocations in those hot paths._

## How to use

```julia
julia> using ConjugateGradients
```

For the conjugate gradient method to solve for `x` in `Ax=b`:

```julia
x, exit_code, num_iters = cg(A, b; kwargs...)
```
```julia
exit_code, num_iters = cg!(A, b, x; kwargs...)
```

For the biconjugate gradient stabilized method:

```julia
x, exit_code, num_iters = bicgstab(A, b; kwargs...)
```
```julia
exit_code, num_iters = bicgstab!(A, b, x; kwargs...)
```

Where `A` must be able to be applied as a function such that `A(b, x)` and the `kwargs` are:
* `tol = 1e-6`: The tolerance of the minimum residual before convergence is accepted.
* `maxIter = 100`: The maximum number of iterations to perform.
* `tolRho = 1e-40`: [`bicgstab` only] The tolerance of `dot(current residual, initial residual)`.
* `precon = nothing`: The preconditioner. The preconditioner must act as an in-place function of the form `f(out, in)`.
* `data = nothing`: The preallocation of the arrays used for solving the system.

### Preallocating

The `data` keyword points to an object containing the preallocated vectors necessary for the functions. If nothing is provided, these vectors will be allocated at each call. The data objects can be created like so:

```julia
CGD = CGData(n, T)
```

```julia
BCGD = BiCGStabData(n, T)
```

Here, `n` is the dimension of the problem and `T` is the type of the elements in the problem (e.g. `Float64`).

### Deciphering the exit code

The `exit_code` can be read with the following function:

```julia
exit_string = reader(exit_code)
```

### A tip for `A` and the preconditioner

The operator `A` and the preconditioner must be expressed as functions. If `A` is a matrix, one can do:

```julia
x, exit_code, num_iters = cg((x,y) -> mul!(x,A,y), b; kwargs...)
```

Another useful representation of `A` is a custom struct. For example, let's consider `(B*C + D)x = b`. Instead of wasting time to build `B*C + D`, we can create a non-allocating version of it.


```julia
struct MyA
    B::SparseMatrixCSC{Float64,Int64}
    C::SparseMatrixCSC{Float64,Int64}
    D::SparseMatrixCSC{Float64,Int64}
    cacheVec::Vector{Float64}
end

function (t::MyA)(out::Vector{Float64}, x::Vector{Float64})
    mul!(t.cacheVec, t.C, x)
    mul!(out, t.B, t.cacheVec)
    mul!(t.cacheVec, t.D, x)
    out .+= t.cacheVec
end

A = MyA(B, C, D, zeros(n))
```
