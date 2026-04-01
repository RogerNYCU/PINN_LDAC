#de los primeros intentos, principalemente para probar la arquitectura de la red, es soft BC

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches
using OptimizationOptimisers, DomainSets, Random
import ModelingToolkit: Interval

# --- 1. PARÁMETROS ---
kh, kc = 1.0, 1.1
rho, cp = 1000.0, 4180.0
alpha = kh / (rho * cp)
Lx, h_val = 2.0, 0.05  # Evitamos confusión con h de la red
um = 0.05
C = (alpha * Lx) / (um * h_val^2)

@parameters x y
@variables Th(..), Tc(..)
Dx = Differential(x)
Dy = Differential(y)
Dyy = Differential(y)^2
u_hat(y) = 1.5 * (1 - (y/h_val)^2) # Perfil normalizado al canal

# --- 2. ECUACIONES Y DOMINIOS ---
eqs = [
    u_hat(y) * Dx(Th(x, y)) ~ C * Dyy(Th(x, y)),
   -u_hat(y) * Dx(Tc(x, y)) ~ C * Dyy(Tc(x, y))
]

bcs = [
    # Entradas (Condiciones de Dirichlet)
    Th(0, y) ~ 1.0, 
    Tc(Lx, y) ~ 0.0, 
    
    # Paredes adiabáticas (Neumann)
    Dy(Th(x, h_val)) ~ 0.0, 
    Dy(Tc(x, -h_val)) ~ 0.0,
    
    # INTERFAZ (y=0): El punto crítico
    Th(x, 0) ~ Tc(x, 0),
    kh * Dy(Th(x, 0)) ~ kc * Dy(Tc(x, 0))
]

domains = [x ∈ Interval(0.0, Lx), y ∈ Interval(-h_val, h_val)]

# --- 3. RED NEURONAL (Arquitectura Burgers-Style) ---
# Usamos tanh porque sigmoid puede "aplanar" demasiado los gradientes en la interfaz
function crear_red()
    return Lux.Chain(
        Lux.Dense(2, 32, Lux.tanh), 
        Lux.Dense(32, 32, Lux.tanh), 
        Lux.Dense(32, 32, Lux.tanh),
        Lux.Dense(32, 1) # Eliminamos sigmoid al final para permitir rango completo
    )
end

chain_h = crear_red()
chain_c = crear_red()

# Cambiamos a GridTraining para mayor estabilidad en la interfaz
strategy = NeuralPDE.GridTraining([0.1, 0.005]) # Malla más fina en 'y' para la interfaz
discretization = PhysicsInformedNN([chain_h, chain_c], strategy)

@named pde_system = PDESystem(eqs, bcs, domains, [x, y], [Th(x, y), Tc(x, y)])
sym_prob = symbolic_discretize(pde_system, discretization)

# --- 4. FUNCIÓN DE PÉRDIDA CON PESOS (Crucial para PINNs) ---
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

function loss_function(θ, p)
    # Pérdida de las ecuaciones (Física)
    pde_loss = sum(l -> sum(abs2, l(θ)), pde_loss_functions)
    
    # Pérdida de condiciones de contorno e INTERFAZ
    # Multiplicamos por 100 para forzar la continuidad térmica
    bc_loss = sum(l -> sum(abs2, l(θ)), bc_loss_functions)
    
    return pde_loss + 100.0 * bc_loss
end

f_ = OptimizationFunction(loss_function, Optimization.AutoZygote())
prob = OptimizationProblem(f_, sym_prob.flat_init_params)

# --- 5. ENTRENAMIENTO HÍBRIDO ---
println("Fase 1: Adam (Estabilizando interfaz)...")
res_adam = Optimization.solve(prob, OptimizationOptimisers.Adam(0.001); maxiters=3000)

println("Fase 2: L-BFGS (Refinamiento físico)...")
prob_lbfgs = OptimizationProblem(f_, res_adam.u)
res_final = Optimization.solve(prob_lbfgs, LBFGS(linesearch = LineSearches.BackTracking()); maxiters=1500)


# --- 6. VISUALIZACIÓN DE RESULTADOS ---
using Plots, Printf

# Crear malla para evaluación
y_plot = LinRange(-h_val, h_val, 50)
x_plot = LinRange(0.0, Lx, 60)

# Extraer parámetros entrenados
θ_final = res_final.u

# Evaluar soluciones en la malla
Th_pred = zeros(length(x_plot), length(y_plot))
Tc_pred = zeros(length(x_plot), length(y_plot))


# Evaluar en toda la malla
for (i, x_val) in enumerate(x_plot)
    for (j, y_val) in enumerate(y_plot)
        Th_pred[i, j] = discretization.phi[1]([x_val, y_val], res_final.u.depvar.Th)[1]
        Tc_pred[i, j] = discretization.phi[2]([x_val, y_val], res_final.u.depvar.Tc)[1]
    end
end

# Gráficas
p1 = heatmap(x_plot, y_plot, Th_pred', xlabel="x", ylabel="y", title="Temperatura Caliente Th(x,y)", 
             palette=:hot, clim=(0.0, 1.0))
p2 = heatmap(x_plot, y_plot, Tc_pred', xlabel="x", ylabel="y", title="Temperatura Fría Tc(x,y)", 
             palette=:cool, clim=(0.0, 1.0))

# Perfiles en interfaz (y=0)
y_interface_idx = argmin(abs.(y_plot .- 0.0))
p3 = plot(x_plot, Th_pred[:, y_interface_idx], label="Th(x,0)", linewidth=2)
plot!(p3, x_plot, Tc_pred[:, y_interface_idx], label="Tc(x,0)", linewidth=2, 
      xlabel="x", ylabel="Temperatura", title="Perfiles en Interfaz (y=0)", legend=:best)

plot(p2, p1,  layout=(2,1), size=(1000, 800))


