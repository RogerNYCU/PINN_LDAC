using NeuralPDE

linear(u, p, t) = cos(t * 2 * pi)
tspan = (0.0, 1.0)
u0 = 0.0
prob = ODEProblem(linear, u0, tspan)

using Lux, Random

rng = Random.default_rng()
Random.seed!(rng, 0)
chain = Chain(Dense(1, 5, σ), Dense(5, 1))
ps, st = Lux.setup(rng, chain) |> Lux.f64

using OptimizationOptimisers

opt = Adam(0.1)
alg = NNODE(chain, opt, init_params = ps)
sol = solve(prob, alg, verbose = true, maxiters = 2000, saveat = 0.01)
using OrdinaryDiffEq, Plots

ground_truth = solve(prob, Tsit5(), saveat = 0.01)

plot(ground_truth, label = "ground truth")
plot!(sol.t, sol.u, label = "pred")