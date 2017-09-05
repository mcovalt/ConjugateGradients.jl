module ConjugateGradients
    include("genericblas.jl")
    include("reader.jl")
    include("cg.jl")
    include("bicgstab.jl")
end
