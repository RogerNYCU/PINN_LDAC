#pinn Hard Boundary Conditions para contraflujo 


using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets,
      OptimizationOptimisers, Plots, Printf, Statistics, Zygote, ComponentArrays
using Random, ForwardDiff, LineSearches, BenchmarkTools

const SEED = 42 

# 1. Fijar la semilla del sistema global
Random.seed!(SEED)

# 2. Crear un generador específico para Lux (si usas Lux)
rng = Random.default_rng()
Random.seed!(rng, SEED)


# --- 1. PARÁMETROS FÍSICOS (Tus valores exactos) ---
Lx, h_phys = 2.0, 0.05
kh, kc = 1.0, 1.1
rho, cp, um = 1000.0, 4180.0, 0.05
Th_in, Tc_in = 80.0, 20.0

# Constantes de Difusión Adimensionales
Ch = (kh * Lx) / (rho * cp * um * h_phys^2)
Cc = (kc * Lx) / (rho * cp * um * h_phys^2)

# --- 2. ARQUITECTURA SIREN (Optimizada para transferencia de calor) ---
# 3 capas de 50 neuronas con 'sin' capturan mejor los gradientes en la interfaz
chain_h = Lux.Chain(Lux.Dense(2, 50, Lux.sin), Lux.Dense(50, 50, Lux.sin), Lux.Dense(50, 1))
chain_c = Lux.Chain(Lux.Dense(2, 50, Lux.sin), Lux.Dense(50, 50, Lux.sin), Lux.Dense(50, 1))

ps_h, st_h = Lux.setup(Random.default_rng(), chain_h)
ps_c, st_c = Lux.setup(Random.default_rng(), chain_c)
θ_init = ComponentVector(h = ps_h, c = ps_c)

# --- 3. MODELOS CON ANSATZ HBC (Estabilidad Mejorada) ---
u_vel(e) = 0.25 * (1.0 - e^2)

# Se mantiene el ansatz, pero se sugiere inicialización cuidadosa
Th_model(x, e, p) = 1.0 + x * chain_h([x, e], p.h, st_h)[1][1]
Tc_model(x, e, p) = 0.0 + (x - 1.0) * chain_c([x, e], p.c, st_c)[1][1]

# --- 4. CÁLCULO DE DERIVADAS ---
function get_derivs(model, x, e, p)
    Tx = ForwardDiff.derivative(vx -> model(vx, e, p), x)
    # Segunda derivada para la difusión (conducción transversal)
    Tee = ForwardDiff.derivative(ve -> ForwardDiff.derivative(vve -> model(x, vve, p), ve), e)
    return Tx, Tee
end

# --- 5. FUNCIÓN DE PÉRDIDA INTEGRADA (Corregida) ---
# --- 1. PREPARACIÓN DE DATOS (Fuera de la pérdida) ---
# Convertimos a Vector para evitar el error de StepRangeLen
# Convertimos a vectores planos para que Zygote no intente diferenciar el constructor del rango
const xs_train = range(0.0, 1.0, length=40)
const es_h = range(0.0, 1.0, length=20)
const es_c = range(-1.0, 0.0, length=20)
# --- 5. FUNCIÓN DE PÉRDIDA ---
function loss_function(θ, _)
    # 1. PDE Canal Caliente (Flujo hacia +x)
    l_h = sum(Iterators.product(xs_train, es_h)) do (x, e)
        Tx, Tee = get_derivs(Th_model, x, e, θ)
        abs2(u_vel(e) * Tx - Ch * Tee)
    end / (40*20)

    # 2. PDE Canal Frío (Contraflujo: velocidad negativa respecto a x)
    l_c = sum(Iterators.product(xs_train, es_c)) do (x, e)
        Tx, Tee = get_derivs(Tc_model, x, e, θ)
        # El signo '-' en u_vel indica que el fluido viaja hacia x=0
        abs2(-u_vel(abs(e)) * Tx - Cc * Tee) 
    end / (40*20)

    # 3. Interfaz (e=0)
    l_int = sum(xs_train) do x
        t0_h = Th_model(x, 0.0, θ)
        t0_c = Tc_model(x, 0.0, θ)
        qh = kh * ForwardDiff.derivative(ve -> Th_model(x, ve, θ), 0.0)
        qc = kc * ForwardDiff.derivative(ve -> Tc_model(x, ve, θ), 0.0)
        abs2(t0_h - t0_c) + abs2(qh - qc)
    end / 40

    # 4. Paredes Adiabáticas
    l_wall = sum(xs_train) do x
        abs2(ForwardDiff.derivative(ve -> Th_model(x, ve, θ), 1.0)) +
        abs2(ForwardDiff.derivative(ve -> Tc_model(x, ve, θ), -1.0))
    end / 40

    return l_h + l_c + 600.0 * l_int + 10.0 * l_wall
end

# --- 6. ENTRENAMIENTO ROBUSTO ---
optf = OptimizationFunction(loss_function, Optimization.AutoZygote())
prob = OptimizationProblem(optf, θ_init)

println(">>> Fase 1: Adam (Ajuste grueso de perfiles)")
@time res = solve(prob, OptimizationOptimisers.Adam(0.0005); maxiters = 1500)

println(">>> Fase 2: LBFGS (Cierre del balance de energía)")
prob2 = OptimizationProblem(optf, res.u)
@time res = solve(prob2, LBFGS(linesearch = BackTracking()); maxiters = 600)
# --- 7. POST-PROCESAMIENTO Y DIAGNÓSTICO (Versión Manual) ---

function post_balance_manual(θ)
    DeltaT = Th_in - Tc_in
    
    # Temperaturas físicas (°C)
    T_h_phys(x, e) = Th_model(x, e, θ) * DeltaT + Tc_in
    T_c_phys(x, e) = Tc_model(x, e, θ) * DeltaT + Tc_in

    # Configuración de integración (Usa 500 puntos para máxima precisión)
    detas = range(0.0, 1.0, length=500)
    de = 1.0 / (length(detas) - 1)
    factor = rho * cp * um * h_phys

    # Canal Caliente: Entrada en x=0, Salida en x=1
    # Multiplicamos por 1.5*(1-e^2) para integrar el flujo de calor real
    e_h_in  = sum(u_vel(e) * T_h_phys(0.0, e) for e in detas) * de
    e_h_out = sum(u_vel(e) * T_h_phys(1.0, e) for e in detas) * de
    Qh = (e_h_in - e_h_out) * factor
    
    # Canal Frío: Entrada en x=1 (Frío), Salida en x=0 (Caliente)
    # Nota: integramos en el dominio del canal frío (e de -1 a 0, usamos abs(e))
    e_c_in  = sum(u_vel(e) * T_c_phys(1.0, -e) for e in detas) * de
    e_c_out = sum(u_vel(e) * T_c_phys(0.0, -e) for e in detas) * de
    Qc = (e_c_out - e_c_in) * factor

    println("\n" * "="^40)
    @printf("RESULTADOS DEL BALANCE (CONTRAFLUJO)\n")
    println("-"^40)
    @printf("Q Cedido (Hot 0->1):  %10.2f W/m\n", Qh)
    @printf("Q Ganado (Cold 1->0): %10.2f W/m\n", Qc)
    @printf("Diferencia Neta:      %10.2f W/m\n", abs(Qh - Qc))
    @printf("Error Balance:        %10.4f %%\n", abs(Qh - Qc)/max(abs(Qh), 1e-9) * 100)
    println("="^40)
end

function visualizacion_manual(θ)
    xs = range(0.0, 1.0, length=50)
    etas_h = range(0.0, 1.0, length=50)
    etas_c = range(-1.0, 0.0, length=50)

    th_map = [Th_model(x, e, θ) for e in etas_h, x in xs]
    tc_map = [Tc_model(x, e, θ) for e in etas_c, x in xs]

    p1 = heatmap(xs, etas_h, th_map, title="Hot Fluid", cmap=:thermal)
    p2 = heatmap(xs, etas_c, tc_map, title="Cold Fluid", cmap=:thermal)
    
    # Perfil transversal en la salida x=1
    y_full = vcat(etas_c, etas_h)
    t_full = vcat([Tc_model(1.0, e, θ) for e in etas_c], [Th_model(1.0, e, θ) for e in etas_h])
    p3 = plot(y_full, t_full, title="Perfil en x=1", xlabel="eta", ylabel="T norm", lw=2, label="PINN")
    vline!([0], color=:black, linestyle=:dash, label="Interfaz")

    display(plot(p1, p2, p3, layout=(3,1), size=(800, 900)))
end

function diagnostico_interfaz_manual(θ)
    xs = range(0.0, 1.0, length=100)
    
    # 1. Continuidad de Temperatura
    Th_int = [Th_model(x, 0.0, θ) for x in xs]
    Tc_int = [Tc_model(x, 0.0, θ) for x in xs]
    p1 = plot(xs, [Th_int Tc_int], label=["Hot" "Cold"], title="T at Interface (eta=0)", lw=2)

    # 2. Continuidad de Flujo (Derivada automática)
    qh = [kh * ForwardDiff.derivative(e -> Th_model(x, e, θ), 0.0) for x in xs]
    qc = [kc * ForwardDiff.derivative(e -> Tc_model(x, e, θ), 0.0) for x in xs]
    p2 = plot(xs, [qh qc], label=["Flux Hot" "Flux Cold"], title="Flux at Interface", lw=2, ls=[:solid :dash])

    display(plot(p1, p2, layout=(2,1), size=(800, 600)))
end

# Ejecución
post_balance_manual(res.u)
visualizacion_manual(res.u)
diagnostico_interfaz_manual(res.u)


