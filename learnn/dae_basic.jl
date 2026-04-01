import DifferentialEquations as DE
import Plots


p = (0.04, 1e4, 3e7, 1e4)

function f2( du, u, p, t)
    k₁, k₂, k₃ = p
    res1 = -k₁*u[1] + k₂ * u[2] * u[3] - du[1]
    res2 = +k₁*u[1] -  k₃* u[2]^2 - k₂ * u[2] * u[3] - du[2]
    res3 = u[1] + u[2] + u[3] - 1.0
    return [res1, res2, res3]
end

u₀ = [1.0, 0, 0]
du₀ = [-0.04, 0.04, 0.0]
tspan = (0.0, 100000.0)

differential_vars = [true, true, false]
prob = DE.DAEProblem(f2, du₀, u₀, tspan, p, differential_vars=differential_vars)

import Sundials
import DiffEqBase
# Explicitly use Brown's initialization algorithm

sol = DE.solve(prob, Sundials.IDA(), reltol=1e-8, abstol=1e-8)

Plots.plot(sol, xscale = :log10, tspan = (1e-6, 1e5), layout = (3, 1))

