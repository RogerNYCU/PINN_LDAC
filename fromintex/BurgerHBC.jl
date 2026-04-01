#No funciona
# Burger con HBC
as=9
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, LineSearches
using OptimizationOptimJL, DomainSets
using Random, DomainSets, Statistics
using Plots

# 1. Configuración del problema (Sin incluir BCs en el sistema para el entrenamiento)
@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# Definimos la ecuación de Burgers
eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - (0.01 / pi) * Dxx(u(t, x)) ~ 0

# Para Hard BCs, las condiciones se definen matemáticamente en la transformación,
# pero las mantenemos aquí para que NeuralPDE sepa las dimensiones del problema.
bcs = [u(0, x) ~ -sin(pi * x), 
       u(t, -1.0) ~ 0.0, 
       u(t, 1.0) ~ 0.0]

domains = [t ∈ Interval(0.0, 1.0), 
           x ∈ Interval(-1.0, 1.0)]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])

# 2. Definición del Ansatz (Transformación Hard BC)
# u_hard(t, x) = A(t, x) + D(t, x) * NN(t, x)
# A(t, x) satisface la condición inicial
# D(t, x) se anula en t=0 y x=±1
function assembly_v(phi, θ, p)
    return (t, x) -> begin
        nn_out = phi([t, x], θ)[1]
        # Término de condición inicial
        A = -sin(pi * x) 
        # Función de distancia que se anula en t=0 y x=1, x=-1
        # Usamos (1-exp(-t)) para el tiempo y (1-x^2) para el espacio
        D = (1 - exp(-t)) * (1 - x^2)
        return A + D * nn_out
    end
end

# Cambiamos tanh por sin (Arquitectura SIREN)
chain = Lux.Chain(
    Lux.Dense(2, 64, sin), 
    Lux.Dense(64, 64, sin), 
    Lux.Dense(64, 64, sin),
    Lux.Dense(64, 1)
)

# Aumentamos los puntos de entrenamiento para capturar el centro
strategy = NeuralPDE.QuasiRandomTraining(3000)

discretization = PhysicsInformedNN(chain, strategy; custom_strategy = assembly_v)

sym_prob = symbolic_discretize(pde_system, discretization)

# 4. Función de Pérdida (¡Solo PDE!)
# Ya no necesitamos bc_loss_functions porque el Ansatz garantiza las BCs
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions

function loss_function(θ, p)
    return sum(l -> mean(abs2, l(θ)), pde_loss_functions)
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())

# 5. Entrenamiento Híbrido
println("Entrenando con Hard BCs (Etapa Adam)...")
prob_adam = OptimizationProblem(f_, sym_prob.flat_init_params)
res_adam = solve(prob_adam, OptimizationOptimisers.Adam(0.005); maxiters = 3000)

println("Entrenando con Hard BCs (Etapa LBFGS)...")
prob_lbfgs = OptimizationProblem(f_, res_adam.u)
res_final = solve(prob_lbfgs, LBFGS(linesearch = BackTracking()); maxiters = 1000)

# 6. Extracción de la solución transformada
# IMPORTANTE: Para visualizar, debemos usar la función de transformación
phi = sym_prob.phi
u_final = assembly_v(phi, res_final.u, nothing)

ts = 0.0:0.01:1.0
xs = -1.0:0.02:1.0
u_pred = [u_final(t, x) for x in xs, t in ts]

# Visualización
p1 = plot(xs, u_pred[:, 1], label="t=0.0 (Exacta por diseño)", lw=3)
plot!(p1, xs, u_pred[:, 51], label="t=0.5", lw=2)
plot!(p1, xs, u_pred[:, 101], label="t=1.0", lw=2)
title!(p1, "Burgers con Hard BCs")

p2 = surface(ts, xs, u_pred, color=:viridis, title="Solución Transformada")
plot(p1, p2, layout=(1,2), size=(1100, 500))




