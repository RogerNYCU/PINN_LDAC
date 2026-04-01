# ============================================================
#  PINN / NNDAE - Tanque térmico (forma ADIMENSIONAL)
# ============================================================

#este codigo si sirve bien

using NeuralPDE, Random, OrdinaryDiffEq, Statistics
using Lux, OptimizationOptimisers
using Plots

# ------------------------------------------------------------
# 1. ECUACIONES NORMALIZADAS
# ------------------------------------------------------------
# u[1] = T*  (temperatura adimensional)
# u[2] = P*  (presión adimensional)
# t    = t*  (tiempo adimensional)

function tanque_termo_norm(du, u, p, t)

    # Ecuación diferencial normalizada
    res1 = du[1] - 1.0

    # Restricción algebraica
    res2 = u[2] - u[1]

    return [res1, res2]
end

# ------------------------------------------------------------
# 2. CONDICIONES INICIALES ADIMENSIONALES
# ------------------------------------------------------------
u₀  = [1.0, 1.0]    # T*(0) = 1, P*(0) = 1
du₀ = [1.0, 0.0]    # dT*/dt* = 1, P* algebraica

# Tiempo adimensional
tspan = (0.0, 1.0)  # puede ser cualquier intervalo

differential_vars = [true, false]

# ------------------------------------------------------------
# 3. RED NEURONAL (PINN)
# ------------------------------------------------------------
rng = Random.default_rng()

chain = Lux.Chain(
    Lux.Dense(1, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 2)
)

# ------------------------------------------------------------
# 4. DEFINICIÓN DEL PROBLEMA DAE
# ------------------------------------------------------------
prob = DAEProblem(
    tanque_termo_norm,
    du₀,
    u₀,
    tspan,
    nothing;
    differential_vars = differential_vars
)

opt = OptimizationOptimisers.Adam(0.01)
alg = NNDAE(chain, opt; autodiff = false)

sol = solve(
    prob,
    alg,
    verbose = true,
    dt = 1 / 100.0,
    maxiters = 2000,
    abstol = 1e-8
)

# ------------------------------------------------------------
# 5. SOLUCIÓN ANALÍTICA NORMALIZADA
# ------------------------------------------------------------
T_star_analitica(t) = 1.0 + t
P_star_analitica(t) = 1.0 + t

# ------------------------------------------------------------
# 6. VISUALIZACIÓN
# ------------------------------------------------------------
t_vals = range(tspan[1], tspan[2], length = 100)

p1 = plot(sol, idxs = 1, label = "PINN", lw = 3,
          title = "Temperatura adimensional T*")
plot!(p1, t_vals, T_star_analitica.(t_vals),
      label = "Analítica", ls = :dash)

p2 = plot(sol, idxs = 2, label = "PINN", lw = 3,
          title = "Presión adimensional P*")
plot!(p2, t_vals, P_star_analitica.(t_vals),
      label = "Analítica", ls = :dash)

plot(p1, p2, layout = (2, 1), xlabel = "Tiempo adimensional t*")

# Parámetros físicos
n  = 1.0
R  = 8.314
Cv = 1.5 * R
V  = 0.01
Q  = 25.0
T0 = 300.0

# Escalas características
tc = (n * Cv * T0) / Q          # tiempo característico [s]
P0 = (n * R * T0) / V           # presión característica [Pa]

# Temperatura real [K]
T_real(t) = T0 * sol(t / tc)[1]

# Presión real [Pa]
P_real(t) = P0 * sol(t / tc)[2]

T_analitica(t) = T0 + (Q / (n * Cv)) * t
P_analitica(t) = (n * R * T_analitica(t)) / V


t_real = range(0.0, 10.0, length = 200)

p1 = plot(
    t_real,
    T_real.(t_real),
    lw = 3,
    label = "PINN",
    title = "Temperatura en el tanque",
    xlabel = "Tiempo [s]",
    ylabel = "Temperatura [K]"
)

plot!(
    p1,
    t_real,
    T_analitica.(t_real),
    ls = :dash,
    lw = 2,
    label = "Analítica"
)
p2 = plot(
    t_real,
    P_real.(t_real),
    lw = 3,
    label = "PINN",
    title = "Presión en el tanque",
    xlabel = "Tiempo [s]",
    ylabel = "Presión [Pa]"
)

plot!(
    p2,
    t_real,
    P_analitica.(t_real),
    ls = :dash,
    lw = 2,
    label = "Analítica"
)
plot(p1, p2, layout = (2,1))







###########################################

# using NeuralPDE, Random, OrdinaryDiffEq, Statistics, Lux, Optimization, OptimizationOptimisers, OptimizationOptimJL, Plots

# # 1. Parámetros físicos
# n, R, V, Q = 1.0, 8.314, 0.01, 25.0
# Cv = 1.5 * R
# T0 = 300.0
# P0 = (n * R * T0) / V
# p = [n, Cv, R, V, Q]

# # --- ESTRATEGIA DE NORMALIZACIÓN ---
# # Definimos valores de referencia para que la red trabaje cerca de 1.0
# T_ref = T0
# P_ref = P0

# function tanque_termo_norm(du, u, p, t)
#     n_p, Cv_p, R_p, V_p, Q_p = p
    
#     # u[1] es T_adimensional, u[2] es P_adimensional
#     T = u[1] * T_ref
#     P = u[2] * P_ref
#     dT_dt = du[1] * T_ref
    
#     # Residuo 1 (Energía): d(T*T_ref)/dt - Q/(n*Cv)
#     # Normalizamos el residuo dividiéndolo por la tasa de cambio esperada (~2.0)
#     res1 = (dT_dt - (Q_p / (n_p * Cv_p))) / 2.0
    
#     # Residuo 2 (Estado): nRT - PV = 0
#     # Normalizamos dividiendo por la energía inicial para que el residuo sea ~0
#     res2 = (n_p * R_p * T - P * V_p) / (n_p * R_p * T_ref)
    
#     return [res1, res2]
# end

# # 2. Configuración de la PINN
# tspan = (0.0, 5.0)
# # Condiciones iniciales normalizadas (T/T_ref = 1, P/P_ref = 1)
# u₀_norm = [1.0, 1.0]
# du₀_norm = [(Q / (n * Cv)) / T_ref, 0.0] 
# differential_vars = [true, false]

# # Red robusta pero simple para evitar las oscilaciones de intentos anteriores
# chain = Lux.Chain(
#     Lux.Dense(1, 16, Lux.tanh),
#     Lux.Dense(16, 16, Lux.tanh),
#     Lux.Dense(16, 2)
# )

# prob = DAEProblem(tanque_termo_norm, du₀_norm, u₀_norm, tspan, p; 
#                   differential_vars = differential_vars)

# # 3. Entrenamiento en dos fases para máxima precisión
# println("Fase 1: Adam para capturar la tendencia...")
# alg1 = NNDAE(chain, OptimizationOptimisers.Adam(0.001); autodiff = false)
# sol = solve(prob, alg1,dt = 1 / 100.0, maxiters = 2000, verbose = true)

# println("Fase 2: LBFGS para ajustar la presión...")
# alg2 = NNDAE(chain, LBFGS(); autodiff = false)
# sol = solve(prob, alg2,dt = 1 / 100.0, maxiters = 800, verbose = true)
# # 4. Post-procesamiento (Des-normalización para graficar)
# t_vals = range(tspan[1], tspan[2], length=100)
# u_pred = [sol(t) for t in t_vals]
# T_pinn = [u[1] * T_ref for u in u_pred]
# P_pinn = [u[2] * P_ref for u in u_pred]

# T_an(t) = T0 + (Q / (n * Cv)) * t
# P_an(t) = (n * R * T_an(t)) / V

# # 5. Gráficas finales
# p1 = plot(t_vals, T_pinn, label="PINN (Normalizada)", lw=3, title="Temperatura [K]")
# plot!(p1, t_vals, T_an.(t_vals), label="Analítica", ls=:dash)

# p2 = plot(t_vals, P_pinn, label="PINN (Normalizada)", lw=3, color=:red, title="Presión [Pa]")
# plot!(p2, t_vals, P_an.(t_vals), label="Analítica", ls=:dash, color=:orange)

# plot(p1, p2, layout=(2,1), size=(800, 600))



aa=0
aa=0
#########################################
#using DifferentialEquations


# using NeuralPDE, Random, OrdinaryDiffEq, Statistics, Lux, OptimizationOptimisers

# using Plots




# function tanque_termo(du, u, p, t)
#     n, Cv, R, V, Q = p
#     # Residuo 1: del orden de 1-10
#     res1 = du[1] - (Q / (n * Cv))
    
#     # Residuo 2: Dividimos por un factor (ej. n*R*T0) para que el error sea relativo
#     res2 = (n * R * u[1] - u[2] * V) / (n * R * 300.0) 
#     return [res1, res2]
# end

# # 2. Parámetros físicos
# n = 1.0        # moles
# R = 8.314      # J/(mol·K)
# Cv = 1.5 * R   # Capacidad calorífica (gas ideal monoatómico)
# V = 0.01       # Volumen en m^3 (10 Litros)
# Q = 25.0      # Calor de entrada en Watts (J/s)
# p = [n, Cv, R, V, Q]
# T0 = 300.0     # 300 K
# P0 = (n * R * T0) / V  # Calculamos P0 para que PV - nRT = 0

# u₀ = [T0, P0]
# du₀ = [2.0046, 1666.6]
# tspan = (0.0, 5.0) # Simulación por 5 segundos
# differential_vars = [true,  false]

# rng = Random.default_rng()
# # Una cadena simple de capas densas
# chain = Lux.Chain(
#     Lux.Dense(1, 32, Lux.tanh),
#     Lux.Dense(32, 32, Lux.tanh),
#     Lux.Dense(32, 32, Lux.tanh),
#     Lux.Dense(32, 2)
# )
# prob = DAEProblem(tanque_termo, du₀, u₀, tspan,p; differential_vars = differential_vars)
# opt = OptimizationOptimisers.Adam(0.01)
# alg = NNDAE(chain, opt; autodiff = false)
# sol = solve(prob, alg, verbose = true, dt = 1 / 100.0, maxiters = 1000, abstol = 1e-8)
# # --- 3. Solución Analítica ---
# # Definimos las funciones basadas en las integrales
# T_analitica(t) = T0 + (Q / (n * Cv)) * t
# P_analitica(t) = (n * R * T_analitica(t)) / V

# # --- 4. Visualización y Comparación ---
# t_vals = range(tspan[1], tspan[2], length=100)
# p1 = plot(sol, idxs=1, label="Numérica (DAE)", title="Temperatura", lw=3, color=:blue)
# plot!(p1, t_vals, T_analitica.(t_vals), label="Analítica", ls=:dash, color=:cyan)


# # Gráfico de Presión
# p2 = plot(sol, idxs=2, label="Numérica (DAE)", title="Presión", lw=3, color=:red)
# plot!(p2, t_vals, P_analitica.(t_vals), label="Analítica", ls=:dash, color=:orange)

# plot(p1, p2, layout=(2,1), xlabel="Tiempo (s)")


# # 6. Graficar resultados
# Plots.plot(sol,  tspan = (1e-6, 5.0), layout = (2, 1))

aa=0


# # function tanque_termo!(du, u, p, t)
# #     n, Cv, R, V, Q = p
# #     T = u[1]
# #     P = u[2]
# #     du[1] = Q / (n * Cv)
# #     du[2] = n * R * T - P * V
# # end

# # # 2. Parámetros físicos
# # n = 1.0        # moles
# # R = 8.314      # J/(mol·K)
# # Cv = 1.5 * R   # Capacidad calorífica (gas ideal monoatómico)
# # V = 0.01       # Volumen en m^3 (10 Litros)
# # Q = 25.0      # Calor de entrada en Watts (J/s)
# # p = [n, Cv, R, V, Q]

# # T0 = 300.0     # 300 K
# # P0 = (n * R * T0) / V  # Calculamos P0 para que PV - nRT = 0
# # u0 = [T0, P0]
# # tspan = (0.0, 15.0) # Simulación por 15 segundos
# # M = [1.0 0.0; 0.0 0.0]


# # f = ODEFunction(tanque_termo!, mass_matrix=M)
# # prob = ODEProblem(f, u0, tspan, p)
# # sol = DifferentialEquations.solve(prob, DifferentialEquations.Rodas5(), reltol = 1e-8, abstol = 1e-8)
