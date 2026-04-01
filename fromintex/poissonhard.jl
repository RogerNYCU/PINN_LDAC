asd=9
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, LineSearches
using Random, Statistics, Plots, DomainSets, OptimizationOptimJL
using BenchmarkTools

# 1. System Configuration
@parameters x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2

# Poisson equation: ∇²u = -sin(πx)sin(πy)
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -sin(pi * x) * sin(pi * y)

# Define domains (0 to 1)
domains = [x ∈ Interval(0.0, 1.0), 
           y ∈ Interval(0.0, 1.0)]

# For HBC, although we don't use bcs in the loss, they are defined for the structure
bcs = [u(0, y) ~ 0.0, u(1, y) ~ 0.0, u(x, 0) ~ 0.0, u(x, 1) ~ 0.0]

@named pde_system = PDESystem(eq, bcs, domains, [x, y], [u(x, y)])

# 2. Hard BC Ansatz Definition
# u(x,y) = x(1-x)y(1-y) * NN(x,y)
function assembly_v(phi, θ, p)
    return (x, y) -> begin
        nn_out = phi([x, y], θ)[1]
        dist_x = x * (1 - x)
        dist_y = y * (1 - y)
        return dist_x * dist_y * nn_out
    end
end

# 3. Neural Network and Discretization
chain = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 1)
)

strategy = NeuralPDE.QuasiRandomTraining(1500)
discretization = PhysicsInformedNN(chain, strategy; custom_strategy = assembly_v)
sym_prob = symbolic_discretize(pde_system, discretization)

# 4. Loss Function (Without BCs, only PDE residual)
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions

function loss_function(θ, p)
    return sum(l -> mean(abs2, l(θ)), pde_loss_functions)
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())

# 5. Training
prob = OptimizationProblem(f_, sym_prob.flat_init_params)

println("Training 2D Poisson with Hard BCs...")
@time res_adam = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 2000)
@time res_final_ph = solve(OptimizationProblem(f_, res_adam.u), LBFGS(); maxiters = 500)

# 6. Solution Visualization with Hard BCs
phi = sym_prob.phi
u_hbc = assembly_v(phi, res_final_ph.u, nothing)
xs = 0.0:0.02:1.0
ys = 0.0:0.02:1.0
u_pred = [u_hbc(x_val, y_val) for x_val in xs, y_val in ys]

p1 = surface(xs, ys, u_pred, title="HBC: u = x(1-x)y(1-y) * NN", color=:viridis)
p2 = contour(xs, ys, u_pred, title="Solution Contours")
plot(p1, p2, layout=(1,2), size=(1000, 450))