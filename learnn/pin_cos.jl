using NeuralPDE, Random, OrdinaryDiffEq, Statistics, Lux, OptimizationOptimisers
function example2(du, u, p, t)
    res1 = cos(2pi * t) - du[1]
    res2 = u[2] + cos(2pi * t) - du[2]
    return [res1, res2]
end
u₀ = [1.0, -1.0]
du₀ = [0.0, 0.0]
tspan = (0.0, 1.0)


differential_vars = [true, false]
prob = DAEProblem(example2, du₀, u₀, tspan; differential_vars = differential_vars)
chain = Lux.Chain(Lux.Dense(1, 15, cos), Lux.Dense(15, 15, sin), Lux.Dense(15, 2))
opt = OptimizationOptimisers.Adam(0.1)
alg = NNDAE(chain, opt; autodiff = false)
sol = solve(prob, alg, verbose = true, dt = 1 / 100.0, maxiters = 4000, abstol = 1e-10)

function example1(du, u, p, t)
    du[1] = cos(2pi * t)
    du[2] = u[2] + cos(2pi * t)
    nothing
end
M = [1.0 0.0; 0.0 0.0]
f = ODEFunction(example1, mass_matrix = M)
prob_mm = ODEProblem(f, u₀, tspan)
ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

using Plots
plot(ground_sol, tspan = tspan, layout = (2, 1), label = "ground truth")
plot!(sol, tspan = tspan, layout = (2, 1), label = "dae with pinns")