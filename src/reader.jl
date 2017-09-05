# Displays exit code information
function reader(exit_code)
    if exit_code == 1
        return "CONVERGED: Provided vector b has zero norm. Returned x as vector of zeros."
    elseif exit_code == 2
        return "CONVERGED: Initial guess of vector x results in sufficient residual."
    elseif exit_code == 30
        return "CONVERGED: Sufficient residual achieved."
    elseif exit_code == 31
        return "CONVERGED: Sufficient residual achieved at first step."
    elseif exit_code == 32
        return "CONVERGED: Sufficient residual achieved at second step."
    elseif exit_code == -11
        return "DID NOT CONVERGE: Rho fell below tolerance."
    elseif exit_code == -12
        return "DID NOT CONVERGE: Omega fell below tolerance."
    elseif exit_code == -13
        return "DID NOT CONVERGE: Zero or infinity alpha."
    elseif exit_code == -2
        return "DID NOT CONVERGE: Maximum iterations reached before convergence."
    else
        return "Invalid error code."
    end
end

export reader