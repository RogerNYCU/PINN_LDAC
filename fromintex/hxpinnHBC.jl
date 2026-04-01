
# HX contraflujo con PINNs y HBC en la interfase, el heat map muestra gradiente, pero mal balance energético
#29/01

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots, Printf, DomainSets
using OptimizationOptimisers, Statistics, LineSearches
using BenchmarkTools



# --- 1. PARÁMETROS FÍSICOS (Consistentes con tu FVM) ---
Lx, h_phys = 2.0, 0.05
rho, cp = 1000.0, 4180.0
kh, kc = 1.0, 1.1
um = 0.05
Th_in, Tc_in = 80.0, 20.0

# Adimensionalización
alpha_h = kh / (rho * cp)
alpha_c = kc / (rho * cp)
Pe_h = (um * h_phys^2) / (alpha_h * Lx)
Pe_c = (um * h_phys^2) / (alpha_c * Lx)

@parameters x eta
@variables Th(..) Tc(..)
Dx = Differential(x)
Deta = Differential(eta)
Deta2 = Differential(eta)^2

# Perfil de velocidad (u*) normalizado
u_star(e) = 6.0 * abs(e) * (1.0 - abs(e))

# --- 2. DEFINICIÓN DEL SISTEMA (Multi-Variable) ---
eqs = [
    u_star(eta) * Dx(Th(x, eta)) ~ (1.0/Pe_h) * Deta2(Th(x, eta)),
   -u_star(eta) * Dx(Tc(x, eta)) ~ (1.0/Pe_c) * Deta2(Tc(x, eta))
]

# Dominios: x ∈ [0,1], eta ∈ [-1, 1] (eta > 0 es Hot, eta < 0 es Cold)
domains = [x ∈ Interval(0.0, 1.0), eta ∈ Interval(-1.0, 1.0)]

# Condiciones de Contorno e Interfaz (Estrategia 3)
bcs = [
    Deta(Th(x, 1.0)) ~ 0.0,   # Pared superior adiabática
    Deta(Tc(x, -1.0)) ~ 0.0,  # Pared inferior adiabática
    Th(x, 0.0) ~ Tc(x, 0.0),  # Continuidad de temperatura
    kh * Deta(Th(x, 0.0)) ~ kc * Deta(Tc(x, 0.0)) # Continuidad de flujo (Estrategia 3)
]

@named pde_system = PDESystem(eqs, bcs, domains, [x, eta], [Th(x, eta), Tc(x, eta)])

# --- 3. RED NEURONAL Y ANSATZ HBC (Estrategia 1) ---
# Usamos dos redes independientes para evitar conflictos de gradiente
input_dim = 2
hidden_dim = 32
chain_h = Lux.Chain(Lux.Dense(input_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, 1))
chain_c = Lux.Chain(Lux.Dense(input_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, 1))

# Ansatz HBC: Fuerza T_in_h = 1 en x=0 y T_in_c = 0 en x=1
function hbc_custom_strategy(phi, θ, p)
    return [
        (x_v, eta_v) -> 1.0 + x_v * phi[1]([x_v, eta_v], θ.depvar.Th)[1],
        (x_v, eta_v) -> 0.0 + (x_v - 1.0) * phi[2]([x_v, eta_v], θ.depvar.Tc)[1]
    ]
end

discretization = PhysicsInformedNN([chain_h, chain_c], 
                                    QuasiRandomTraining(1000); 
                                    custom_strategy = hbc_custom_strategy)

# Discretización simbólica
sym_prob = symbolic_discretize(pde_system, discretization)

# --- 4. CONSTRUCCIÓN MANUAL DE LA FUNCIÓN DE PÉRDIDA ---
# Esto corrige el error de "PINNRepresentation has no field f"
pde_loss_f = sym_prob.loss_functions.pde_loss_functions
bc_loss_f = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    @printf("Pérdida Total: %.6e\n", l)
    return false
end

loss_func = OptimizationFunction((θ, p) -> begin
    # Pérdida de las ecuaciones (PDE)
    l_pde = sum(f -> mean(abs2, f(θ)), pde_loss_f)
    
    # Separamos las BCs para darles más peso a la interfase (índices 3 y 4 en bcs)
    # 1: Adiabática H, 2: Adiabática C, 3: Continuidad T, 4: Continuidad Flujo
    l_bc_ext = mean(abs2, bc_loss_f[1](θ)) + mean(abs2, bc_loss_f[2](θ))
    l_interface = mean(abs2, bc_loss_f[3](θ)) + mean(abs2, bc_loss_f[4](θ))
    
    # El factor 50.0 fuerza el balance de energía
    return l_pde + l_bc_ext + 50.0 * l_interface 
end, Optimization.AutoZygote())

opt_prob = OptimizationProblem(loss_func, sym_prob.flat_init_params)

# --- 5. ENTRENAMIENTO ---
println("Fase 1: Adam para estabilización...")
res = solve(opt_prob, OptimizationOptimisers.Adam(0.008); maxiters = 500)

# Refinamiento agresivo
println("Fase 2: LBFGS Agresivo...")
opt_prob2 = OptimizationProblem(loss_func, res.u)
res = solve(opt_prob2, LBFGS(
    linesearch = BackTracking() # Ayuda a no saltarse el mínimo
); 
    maxiters = 2000, 
    # Tolerancias más estrictas para forzar el balance
    gtol = 1e-9, 
    callback = callback
)


# --- 6. RESULTADOS Y POST-PROCESADO ---
phi = discretization.phi
T_h_f(x_v, e_v) = 1.0 + x_v * phi[1]([x_v, e_v], res.u.depvar.Th)[1]
T_c_f(x_v, e_v) = 0.0 + (x_v - 1.0) * phi[2]([x_v, e_v], res.u.depvar.Tc)[1]

# Visualización comparativa
xs = 0.0:0.05:1.0
etas_h = 0.0:0.05:1.0
etas_c = -1.0:0.05:0.0

th_map = [T_h_f(xv, ev) for ev in etas_h, xv in xs]
tc_map = [T_c_f(xv, ev) for ev in etas_c, xv in xs]

p1 = heatmap(xs, etas_h, th_map, title="Canal Caliente (HBC)", cmap=:thermal)
p2 = heatmap(xs, etas_c, tc_map, title="Canal Frío (HBC)", cmap=:thermal)
plot(p1, p2, layout=(2,1))

# --- 7. CÁLCULO DE BALANCE DE ENERGÍA (Watts/m) ---

function calcular_balance_pinn(res, discretization, params)
    phi = discretization.phi
    u_params = res.u
    
    # Extraer parámetros físicos
    rho, cp, um, h = params[:rho], params[:cp], params[:um], params[:h]
    Th_in, Tc_in = params[:Th_in], params[:Tc_in]
    DeltaT = Th_in - Tc_in
    
    # Funciones de temperatura desnormalizadas (°C)
    T_h_phys(x, e) = (1.0 + x * phi[1]([x, e], u_params.depvar.Th)[1]) * DeltaT + Tc_in
    T_c_phys(x, e) = (0.0 + (x - 1.0) * phi[2]([x, e], u_params.depvar.Tc)[1]) * DeltaT + Tc_in

    # Configuración de integración (Trapezoidal sobre eta)
    n_steps = 200
    detas = range(0.0, 1.0, length=n_steps)
    de = 1.0 / (n_steps - 1)
    
    # Factor de conversión: rho * cp * um * h_canal
    # Nota: h_phys es la altura de un solo canal (0.05)
    factor = rho * cp * um * h 

    # --- Canal Caliente (Hot) ---
    # Entrada en x=0 (Sabemos que es Th_in por HBC, pero integramos para validar)
    flux_h_in = sum(u_star(e) * T_h_phys(0.0, e) for e in detas) * de
    # Salida en x=1
    flux_h_out = sum(u_star(e) * T_h_phys(1.0, e) for e in detas) * de
    
    Q_hot = (flux_h_in - flux_h_out) * factor

    # El canal frío entra por x=1 y sale por x=0
    # Calor ganado = Energía que SALE(x=0) - Energía que ENTRA(x=1)
    flux_c_out = sum(u_star(e) * T_c_phys(0.0, -e) for e in detas) * de
    flux_c_in  = sum(u_star(e) * T_c_phys(1.0, -e) for e in detas) * de
    Q_cold = (flux_c_out - flux_c_in) * factor

    # --- Reporte ---
    error_rel = abs(Q_hot - Q_cold) / max(Q_hot, Q_cold) * 100

    println("\n" * "="^40)
    println("      REPORTE DE BALANCE TÉRMICO PINN")
    println("="^40)
    @printf("Q Cedido (Canal Caliente):  %10.2f W/m\n", Q_hot)
    @printf("Q Ganado (Canal Frío):     %10.2f W/m\n", Q_cold)
    println("-"^40)
    @printf("Error de Balance:           %10.4f %%\n", error_rel)
    println("="^40)
    
    return Q_hot, Q_cold
end

# Definir diccionario de parámetros para la función
phys_params = Dict(
    :rho => rho, :cp => cp, :um => um, :h => h_phys,
    :Th_in => Th_in, :Tc_in => Tc_in
)

# Ejecutar cálculo
Qh, Qc = calcular_balance_pinn(res, discretization, phys_params)

