# Viga con carga, PINN con SBC
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, LineSearches, DomainSets
using OptimizationOptimJL
using Plots
using BenchmarkTools

# 1. Configuración del Espacio y Variables
@parameters x
@variables u(..)
Dx = Differential(x)
D2x = Differential(x)^2
D3x = Differential(x)^3
D4x = Differential(x)^4

L = 1.0 # Longitud de la viga
q = 1.0 # Carga uniforme

# 2. Definición de la Ecuación y BCs (Soft)
# Ecuación de Euler-Bernoulli: u'''' = q/EI
eq = D4x(u(x)) ~ q

bcs = [
    u(0.0) ~ 0.0,        # Deflexión cero en x=0
    Dx(u(0.0)) ~ 0.0,    # Pendiente cero en x=0 (Clamped)
    D2x(u(L)) ~ 0.0,     # Momento cero en x=L (Free)
    D3x(u(L)) ~ 0.0      # Cortante cero en x=L (Free)
]

domains = [x ∈ Interval(0.0, L)]

@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])

# 3. Red Neuronal
# Para ecuaciones de 4to orden, necesitamos una red con suficiente profundidad
chain = Lux.Chain(
    Lux.Dense(1, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 1)
)

# Estrategia de entrenamiento
strategy = NeuralPDE.GridTraining(0.05)
discretization = PhysicsInformedNN(chain, strategy)
sym_prob = symbolic_discretize(pde_system, discretization)

# 4. Función de Pérdida con Pesos (Crucial para Soft BCs)
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

function loss_function(θ, p)
    pde_loss = sum(l -> sum(abs2, l(θ)), pde_loss_functions)
    bc_loss = sum(l -> sum(abs2, l(θ)), bc_loss_functions)
    # λ = 100.0 ayuda a que la viga no se "despegue" de la pared en x=0
    return pde_loss + 100.0 * bc_loss
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())

# 5. Entrenamiento Híbrido
prob = OptimizationProblem(f_, sym_prob.flat_init_params)

println("Entrenando Etapa 1 (Adam)...")
@time res_adam = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 3000)

println("Entrenando Etapa 2 (LBFGS)...")
@time prob_lbfgs = OptimizationProblem(f_, res_adam.u)
@time res_final = solve(prob_lbfgs, LBFGS(linesearch = BackTracking()); maxiters = 1000)

# 6. Visualización y Validación Analítica
# Solución exacta: u(x) = (q*x^2 / 24*EI) * (6L^2 - 4Lx + x^2)
u_exact(x) = (q * x^2 / 24.0) * (6*L^2 - 4*L*x + x^2)

xs = 0.0:0.01:L
phi = sym_prob.phi
u_pred = [phi([x], res_final.u)[1] for x in xs]
u_true = [u_exact(x) for x in xs]

plot(xs, u_true, label="Exacta (Bernoulli)", lw=3, ls=:dash, color=:black)
plot!(xs, u_pred, label="PINN (Soft BC)", lw=2, color=:red)
title!("Deflexión de Viga Cantilever")
xlabel!("x (Longitud)")

ylabel!("u(x) (Deflexión)")

