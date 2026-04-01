using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets
using OptimizationOptimisers, Statistics
using BenchmarkTools, Plots
# --- 1. PARÁMETROS ---
# --- 1. PARÁMETROS FÍSICOS ---
Lx, h_phys = 5.0, 0.05
kh, kc = 1.0, 1.1
rho, cp, um = 1000.0, 4180.0, 0.05
Th_in, Tc_in = 80.0, 20.0

# Constantes Normalizadas para la PDE
C_h = ((kh / (rho * cp)) * Lx) / (um * h_phys^2)
C_c = ((kc / (rho * cp)) * Lx) / (um * h_phys^2)

@parameters x eta
@variables T1(..), T2(..) # Dos redes: T1 para caliente, T2 para frío
Dx = Differential(x)
Deta = Differential(eta)
Detaeta = Differential(eta)^2

# --- 2. DOMINIOS Y ECUACIONES ---
# Definimos el dominio total, pero cada variable actuará en su mitad
domains = [x ∈ Interval(0.0, 1.0), eta ∈ Interval(-1.0/kc, 1.0/kh)]

eqs = [
    # PDE Canal Caliente (eta > 0)
    u_vel(eta) * Dx(T1(x, eta)) ~ C_h * Detaeta(T1(x, eta)),
    
    # PDE Canal Frío (eta < 0)
    u_vel(eta) * Dx(T2(x, eta)) ~ C_c * Detaeta(T2(x, eta))
]

# --- 3. CONDICIONES DE CONTORNO Y ACOPLAMIENTO ---
bcs = [
    # Entrada
    T1(0.0, eta) ~ 1.0, # Caliente entra a 1
    T2(0.0, eta) ~ 0.0, # Frío entra a 0
    
    # Aislamiento en las paredes externas
    Deta(T1(x, 1.0/kh)) ~ 0.0,
    Deta(T2(x, -1.0/kc)) ~ 0.0,
    
    # ACOPLAMIENTO EN LA INTERFAZ (eta = 0)
    # 1. Continuidad de temperatura
    T1(x, 0.0) ~ T2(x, 0.0),
    
    # 2. Continuidad del flujo de calor (Ley de Fourier)
    # k_h * dT1/deta = k_c * dT2/deta
    kh * Deta(T1(x, 0.0)) ~ kc * Deta(T2(x, 0.0))
]

# --- 4. REDES NEURONALES (Una para cada variable) ---
# Al pasar una lista de cadenas, NeuralPDE asigna una a T1 y otra a T2
input_dims = 2
chains = [
    Lux.Chain(Lux.Dense(input_dims, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin),  Lux.Dense(32, 1)),
    Lux.Chain(Lux.Dense(input_dims, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin),  Lux.Dense(32, 1))
]

# --- 5. DISCRETIZACIÓN ---
strategy = NeuralPDE.QuasiRandomTraining(1500)
discretization = PhysicsInformedNN(chains, strategy)

@named pde_system = PDESystem(eqs, bcs, domains, [x, eta], [T1(x, eta), T2(x, eta)])
prob = discretize(pde_system, discretization)
# --- 6. OPTIMIZACIÓN ---
res = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 500)
# Refinamiento opcional con LBFGS
prob2 = remake(prob, u0 = res.u)
res = solve(prob2, LBFGS(); maxiters = 200)

using Plots, Printf

function graficar_sistema_acoplado(res, discretization, kh, kc)
    # 1. Definir rangos de visualización
    xs = 0.0:0.01:1.0
    # Creamos dos rangos para eta y los unimos
    etas_h = range(0.0, 1.0/kh, length=100)
    etas_c = range(-1.0/kc, 0.0, length=100)
    
    # Obtener las funciones de las redes neuronales
    # discretization.phi[1] es para T1 (caliente), [2] para T2 (frío)
    T1_func = discretization.phi[1]
    T2_func = discretization.phi[2]
    
    # 2. Evaluar las redes en sus dominios correspondientes
    # Nota: Usamos los parámetros optimizados res.u
    # En sistemas multivariable, NeuralPDE organiza los parámetros en el vector u
    
    # Para el canal caliente (T1)
  # 1. Obtenemos los parámetros separados para cada red
# NeuralPDE suele agruparlos en una estructura que podemos extraer así:
params_t1 = res.u.depvar.T1
params_t2 = res.u.depvar.T2

# 2. Evaluamos usando los parámetros específicos de cada red
# Canal caliente (T1)
T_map_h = [T1_func([xv, ev], params_t1)[1] for ev in etas_h, xv in xs]

# Canal frío (T2)
T_map_c = [T2_func([xv, ev], params_t2)[1] for ev in etas_c, xv in xs]
    
    # 3. Concatenar los mapas (Frío abajo, Caliente arriba)
    T_total = vcat(T_map_c, T_map_h)
    etas_total = vcat(collect(etas_c), collect(etas_h))
    
    # 4. Generar el Heatmap
    p = heatmap(xs, etas_total, T_total,
                title = "PINN Acoplada: 2 Redes Neuronales",
                xlabel = "Posición x (Normalizada)",
                ylabel = "Coordenada eta (Normalizada)",
                cmap = :thermal,
                colorbar_title = "Temperatura T",
                right_margin = 10Plots.mm)
    
    # Añadir una línea en eta=0 para marcar la interfaz
    hline!([0.0], color=:white, linestyle=:dash, label="Interfaz", lw=2)
    
    return p
end

# Ejecutar la gráfica
p_final = graficar_sistema_acoplado(res, discretization, kh, kc)