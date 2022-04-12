import LinearAlgebra: BLAS, BlasFloat, norm

genblas_dot(x::Vector{T}, y::Vector{T}) where {T<:BlasFloat} = BLAS.dot(x, y)
genblas_dot(x, y) = dot(x,y)
genblas_scal!(a::T, x::Vector{T}) where {T<:BlasFloat} = BLAS.scal!(length(x), a, x, 1)
genblas_scal!(a, x) = x .*= a
genblas_axpy!(a::T, x::Vector{T}, y::Vector{T}) where {T<:BlasFloat} = BLAS.axpy!(a, x, y)
genblas_axpy!(a, x, y) = y .+= a.*x
genblas_nrm2(x::Vector{T}) where {T<:BlasFloat} = BLAS.nrm2(length(x), x, 1)
genblas_nrm2(x) = norm(x)
