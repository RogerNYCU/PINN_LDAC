import DifferentialEquations as DE
import Plots

function example1(du, u, p, t)
    du[1] = cos(2pi * t)
    du[2] = u[2] + cos(2pi * t)
    nothing
end
M = [1.0 0.0; 0.0 0.0]
f = ODEFunction(example1, mass_matrix = M)
prob_mm = ODEProblem(f, u₀, tspan)
ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

