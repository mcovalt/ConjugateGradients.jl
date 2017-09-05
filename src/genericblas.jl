genblas_dot{T<:Base.LinAlg.BlasFloat}(x::Vector{T}, y::Vector{T}) = BLAS.dot(x, y)
genblas_dot(x, y) = dot(x,y)
genblas_scal!{T<:Base.LinAlg.BlasFloat}(a::T, x::Vector{T}) = BLAS.scal!(length(x), a, x, 1)
genblas_scal!(a, x) = x .*= a
genblas_axpy!{T<:Base.LinAlg.BlasFloat}(a::T, x::Vector{T}, y::Vector{T}) = BLAS.axpy!(a, x, y)
genblas_axpy!(a, x, y) = y .+= a.*x
genblas_nrm2{T<:Base.LinAlg.BlasFloat}(x::Vector{T}) = BLAS.nrm2(length(x), x, 1)
genblas_nrm2(x) = norm(x)