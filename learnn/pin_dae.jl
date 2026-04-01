
using NeuralPDE, Random, OrdinaryDiffEq, Statistics, Lux, OptimizationOptimisers
using Pkg
Pkg.add("NeuralPDE")
Pkg.add("Random")
Pkg.add("OrdinaryDiffEq")
Pkg.add("Statistics")
Pkg.add("Lux")
Pkg.add("OptimizationOptimisers")


# function example2(du, u, p, t)
#     res1 = cos(2pi * t) - du[1]
#     res2 = u[2] + cos(2pi * t) - du[2]
#     return [res1, res2]
# end
# 1. Definición del problema de Robertson (DAE)
# El formato debe ser f(du, u, p, t) = 0
function robertson_dae(du, u, p, t)
    res1 = -0.04u[1] + 1e4 * u[2] * u[3] - du[1]
    res2 = +0.04u[1] - 3e7 * u[2]^2 - 1e4 * u[2] * u[3] - du[2]
    res3 = u[1] + u[2] + u[3] - 1.0
    
    return [res1, res2, res3]
end

# 2. Condiciones iniciales consistentes (extraídas de la imagen)
u₀ = [1.0, 0.0, 0.0]
du₀ = [-0.04, 0.04, 0.0]
limitet=1.0
tspan = (0.0, limitet) # Se recomienda un rango corto para PINNs antes de ir a 1e5

differential_vars = [true, true, false]


# Una cadena simple de capas densas
chain = Chain(
    Dense(1, 64, tanh), 
    Dense(64, 64, tanh), 
    Dense(64, 3)
)
prob = DAEProblem(robertson_dae, du₀, u₀, tspan; differential_vars = differential_vars)

opt = OptimizationOptimisers.Adam(0.001)
alg = NNDAE(chain, opt; autodiff = false)
sol = solve(prob, alg, verbose = true, dt = 1 / 100.0, maxiters = 4000, abstol = 1e-8)


function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    nothing
end
M = [1.0 0 0
     0 1.0 0
     0 0 0]
f = ODEFunction(rober, mass_matrix = M)
prob_mm = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, limitet), (0.04, 3e7, 1e4))
ground_sol = solve(prob_mm, Rodas5(), reltol = 1e-8, abstol = 1e-8)

using Plots
Plots.plot(ground_sol, xscale = :log10, tspan = (1e-6, limitet), layout = (3, 1),label = "ground truth")
#plot!(sol, xscale = :log10, tspan = (1e-6, limitet), layout = (3, 1),label = "pinn ")

limitet