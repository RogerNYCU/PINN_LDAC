# Burger's equation better version with PINN SBC

xas=87
 using NeuralPDE , Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches, Zygote
using OptimizationOptimisers
using Random, DomainSets
using BenchmarkTools
using Plots

# 1. Problem Configuration
rng = Random.default_rng()
@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Burgers Equation
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

bcs = [u(0, x) ~ -sin(pi * x), 
    u(t, -1.0) ~ 0.0, 
    u(t, 1.0) ~ 0.0]

domains = [t ∈ Interval(0.0, 1.0), 
        x ∈ Interval(-1.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# 2. Neural Network: Increased depth to capture the "shock"
chain = Lux.Chain(
    Lux.Dense(2, 20, Lux.tanh), 
    Lux.Dense(20, 20, Lux.tanh), 
    Lux.Dense(20, 20, Lux.tanh), 
    Lux.Dense(20, 20, Lux.tanh), 
    Lux.Dense(20, 1)
)

strategy = NeuralPDE.GridTraining(0.05) 
discretization = PhysicsInformedNN(chain, strategy)
sym_prob = symbolic_discretize(pde_system, discretization)
phi = sym_prob.phi # Extract the network function

# 3. Loss function with aggressive boundary penalization
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

function loss_function(θ, p)
    pde_loss = sum(l -> sum(abs2, l(θ)), pde_loss_functions)
    bc_loss = sum(l -> sum(abs2, l(θ)), bc_loss_functions)
    # λ = 100 forces the network to not deviate from the zero axis
    return pde_loss + 100.0 * bc_loss
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())

# 4. Hybrid Training (Adam -> LBFGS)
println("Training Stage 1 (Adam)...")
prob_adam = OptimizationProblem(f_, sym_prob.flat_init_params)
@time res_adam = solve(prob_adam, OptimizationOptimisers.Adam(0.002); maxiters = 1000) #3k

println("Training Stage 2 (LBFGS)...")
prob_lbfgs = OptimizationProblem(f_, res_adam.u)
@time res_final = solve(prob_lbfgs, LBFGS(linesearch = BackTracking()); maxiters = 1000)

# 5. Visualization and Comparison
ts = 0.0:0.01:1.0
xs = -1.0:0.02:1.0
u_pred = [phi([t, x], res_final.u)[1] for x in xs, t in ts]

# Approximate Analytical Solution (For t=0 it is simply -sin(pi*x))
u_exact_t0 = -sin.(pi .* xs)

p1 = plot(xs, u_pred[:, 1], label="PINN t=0.0", lw=3, color=:blue, xlabel="x", ylabel="u(t,x)")
plot!(p1, xs, u_exact_t0, label="Exact t=0.0", ls=:dash, color=:black)
plot!(p1, xs, u_pred[:, 51], label="PINN t=0.5", lw=2, color=:red)
plot!(p1, xs, u_pred[:, 101], label="PINN t=1.0", lw=2, color=:green)
title!(p1, "Scale and Shape Comparison")

p2 = surface(ts, xs, u_pred, title="PINN Solution (Viscosity 0.01/π)", 
          xlabel="t", ylabel="x", color=:magma)

plot(p1, p2, layout=(1,2), size=(1000, 500))