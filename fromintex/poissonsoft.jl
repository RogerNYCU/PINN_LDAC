using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, LineSearches
using Plots, Random, Statistics, DomainSets, OptimizationOptimJL, BenchmarkTools

# 1. Variables and Parameters
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# Poisson equation: ∇²u = f(x,y)
# We use a source function that generates a "mountain" in the center
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# 2. Boundary Conditions (Soft BCs)
# Define u=0 on the four walls of the square [0,1]x[0,1]
bcs = [
    u(0, y) ~ 0.0, u(1, y) ~ 0.0, # Edges in x
    u(x, 0) ~ 0.0, u(x, 1) ~ 0.0  # Edges in y
]

domains = [x ∈ Interval(0.0, 1.0), 
           y ∈ Interval(0.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

# 3. Neural Network
# A simple network with 2 inputs (x,y) and 1 output
chain = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 1)
)

# Quasi-random training to well cover the XY plane
strategy = NeuralPDE.QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy)
sym_prob = symbolic_discretize(pde_system, discretization)

# 4. Loss Function
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

function loss_function(θ, p)
    pde_loss = sum(l -> sum(abs2, l(θ)), pde_loss_functions)
    bc_loss = sum(l -> sum(abs2, l(θ)), bc_loss_functions)
    return pde_loss + 50.0 * bc_loss # Moderate weight for boundaries
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())

# 5. Hybrid Optimization
prob = OptimizationProblem(f_, sym_prob.flat_init_params)

println("Training 2D Poisson...")
@time res_adam = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 2000)
prob_lbfgs = OptimizationProblem(f_, res_adam.u)
@time res_final_poiSB = solve(prob_lbfgs, LBFGS(); maxiters = 500)

# 6. Visualization
xs = 0.0:0.02:1.0
ys = 0.0:0.02:1.0
phi_ps = sym_prob.phi
u_pred_ps = [phi_ps([x, y], res_final_poiSB.u)[1] for x in xs, y in ys]

surface(xs, ys, u_pred_ps, title="PINN Solution: 2D Poisson", xlabel="x", ylabel="y", zlabel="u(x,y)")
