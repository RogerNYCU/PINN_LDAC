# Viga con carga,no funciona

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, LineSearches
using DomainSets, Plots
using Random, Statistics, Plots
using BenchmarkTools
@parameters x
@variables u(..)
Dx = Differential(x); D2x = Differential(x)^2; D3x = Differential(x)^3; D4x = Differential(x)^4

L = 1.0
P = -0.1 # Carga puntual hacia abajo en el extremo libre
EI = 1.0

# Ecuación de la viga sin carga distribuida
eq = D4x(u(x)) ~ 0.0

# Condiciones de Contorno para carga puntual en el extremo (Soft para las fuerzas)
bcs = [
    u(0.0) ~ 0.0, 
    Dx(u(0.0)) ~ 0.0,
    D2x(L) ~ 0.0,      # Momento flector cero en el extremo
    D3x(L) ~ P/EI      # Cortante igual a la carga puntual P
]

domains = [x ∈ Interval(0.0, L)]
@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])

# Ansatz HBC: Garantiza u(0)=0 y u'(0)=0
function cantilever_hbc(phi, θ, p)
    return (x) -> (x^2) * phi([x], θ)[1]
end

# Red Neuronal más robusta (64 neuronas)
chain = Lux.Chain(Lux.Dense(1, 64, Lux.tanh), Lux.Dense(64, 64, Lux.tanh), Lux.Dense(64, 1))

discretization = PhysicsInformedNN(chain, NeuralPDE.GridTraining(0.02); custom_strategy = cantilever_hbc)
sym_prob = symbolic_discretize(pde_system, discretization)

# Pérdida: Solo PDE + las BCs de fuerza (las de posición ya están en el Ansatz)
function loss_function(θ, p)
    pde_loss = mean(abs2, sym_prob.loss_functions.pde_loss_functions[1](θ))
    # Solo necesitamos las BCs de momento y cortante (índices 3 y 4)
    bc_loss = mean(abs2, sym_prob.loss_functions.bc_loss_functions[3](θ)) + 
              mean(abs2, sym_prob.loss_functions.bc_loss_functions[4](θ))
    return pde_loss + 100.0 * bc_loss
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
@time res_adam = solve(OptimizationProblem(f_, sym_prob.flat_init_params), Adam(0.002); maxiters = 3000)
@time res_final = solve(OptimizationProblem(f_, res_adam.u), Optimization.LBFGS(); maxiters = 1000)

# Validación Analítica (Carga puntual en el extremo)
# u(x) = (P*x^2 / (6*EI)) * (3L - x)
u_exact(x) = (P * x^2 / (6.0 * EI)) * (3*L - x)

xs = 0.0:0.01:L
u_pinn = [cantilever_hbc(sym_prob.phi, res_final.u, nothing)(x_val) for x_val in xs]
u_true = [u_exact(x_val) for x_val in xs]

plot(xs, u_true, label="Analítica (Puntual)", lw=4, color=:black, alpha=0.3)
plot!(xs, u_pinn, label="PINN HBC", lw=2, color=:red)
title!("Viga con Carga Puntual en Extremo")