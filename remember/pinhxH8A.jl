#usa dos redes, 507K calor, 43% error, gradientes buenos. 

#algoritmo para balance de energia 
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets,
      OptimizationOptimisers, Plots, Printf, Statistics, Zygote, ComponentArrays
using Random, ForwardDiff, BenchmarkTools, LineSearches
# Principales diferencias entre los dos códigos:

# ## 1. **Framework de discretización**
# - **Código 1**: Usa `NeuralPDE.jl` con `PhysicsInformedNN` y discretización simbólica automática
# - **Código 2**: Implementación manual usando `ForwardDiff` para cálculo de derivadas

# ## 2. **Arquitectura de redes**
# - **Código 1**: Dense estándar con `tanh` (32 neuronas)
# - **Código 2**: SIREN con `sin` (50 neuronas) - mejor para capturar gradientes

# ## 3. **Función de pérdida**
# - **Código 1**: Pérdidas automáticas extraídas de `sym_prob.loss_functions`
# - **Código 2**: Pérdida manual construida con bucles explícitos sobre rejilla de entrenamiento

# ## 4. **Generación de derivadas**
# - **Código 1**: Automática vía `Differential()` y symbolic discretization
# - **Código 2**: Manual con `ForwardDiff.derivative()` en cada evaluación

# ## 5. **Condiciones de contorno (Ansatz HBC)**
# - **Código 1**: Defino en `hbc_custom_strategy` como funciones anónimas
# - **Código 2**: Incorporadas directamente en `Th_model` y `Tc_model`

# ## 6. **Entrenamiento**
# - **Código 1**: Adam (0.008, 2 iter) + LBFGS (2 iter)
# - **Código 2**: Adam (0.001, 1500 iter) + LBFGS (800 iter) con **peso 500x en interfaz**

# ## 7. **Balance energético**
# - **Código 1**: Función `calcular_balance_pinn()` genérica
# - **Código 2**: Función `post_balance_manual()` más optimizada + diagnóstico detallado de interfaz

# **Resumen clave**: Código 2 es más manual pero con mejor control sobre la convergencia de balance (control de pesos en loss), mejor arquitectura (SIREN) y más diagnóstico.

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

# --- 3. MODELOS CON ANSATZ HBC (Garantizan condiciones de entrada) ---
# Perfil de velocidad parabólico clásico (Laminar)
u_vel(e) = 1.5 * (1.0 - e^2)

# Th(x, e): Th(0, e) = 1.0 (Entrada caliente)
Th_model(x, e, p) = 1.0 + x * chain_h([x, e], p.h, st_h)[1][1]
# Tc(x, e): Tc(1, e) = 0.0 (Entrada fría en contraflujo)
Tc_model(x, e, p) = 0.0 + (x - 1.0) * chain_c([x, e], p.c, st_c)[1][1]

# --- 4. CÁLCULO DE DERIVADAS (Manual con ForwardDiff) ---
function get_derivs(model, x, e, p)
    # Advección (x) y Difusión (e)
    Tx = ForwardDiff.derivative(vx -> model(vx, e, p), x)
    Tee = ForwardDiff.derivative(ve -> ForwardDiff.derivative(vve -> model(x, vve, p), ve), e)
    return Tx, Tee
end

# --- 5. FUNCIÓN DE PÉRDIDA INTEGRADA ---
# Rejilla de entrenamiento: 40x20 por canal para asegurar resolución
xs_train = range(0.0, 1.0, length=40)
es_h = range(0.0, 1.0, length=20)
es_c = range(-1.0, 0.0, length=20)

function loss_function(θ, _)
    # 1. PDE Canal Caliente
    l_h = sum(Iterators.product(xs_train, es_h)) do (x, e)
        Tx, Tee = get_derivs(Th_model, x, e, θ)
        abs2(u_vel(e) * Tx - Ch * Tee)
    end / (40*20)

    # 2. PDE Canal Frío (Contraflujo: velocidad negativa respecto a x)
    l_c = sum(Iterators.product(xs_train, es_c)) do (x, e)
        Tx, Tee = get_derivs(Tc_model, x, e, θ)
        abs2(-u_vel(e) * Tx - Cc * Tee) # Notar el signo '-' por contraflujo
    end / (40*20)

    # 3. INTERFAZ (e=0): El "Pegamento" de las redes
    l_int = sum(xs_train) do x
        t0_h = Th_model(x, 0.0, θ)
        t0_c = Tc_model(x, 0.0, θ)
        
        # Continuidad de Flujo: kh * dTh/de = kc * dTc/de
        qh = kh * ForwardDiff.derivative(ve -> Th_model(x, ve, θ), 0.0)
        qc = kc * ForwardDiff.derivative(ve -> Tc_model(x, ve, θ), 0.0)
        
        abs2(t0_h - t0_c) + abs2(qh - qc)
    end / 40

    # 4. PAREDES ADIABÁTICAS (Bordes externos)
    l_wall = sum(xs_train) do x
        abs2(ForwardDiff.derivative(ve -> Th_model(x, ve, θ), 1.0)) +
        abs2(ForwardDiff.derivative(ve -> Tc_model(x, ve, θ), -1.0))
    end / 40

    # Peso agresivo en la interfaz para cerrar el balance
    return l_h + l_c + 500.0 * l_int + 10.0 * l_wall
end
callback = function (p, l)
    if !@isdefined(iter_count) global iter_count = 0 end
    global iter_count += 1
    if iter_count % 100 == 0 println("Iteración: $iter_count | Pérdida: $l") end
    return false
end
# --- 6. ENTRENAMIENTO EN DOS ETAPAS ---
optf = OptimizationFunction(loss_function, Optimization.AutoZygote())
prob = OptimizationProblem(optf, θ_init)

println(">>> Fase 1: Adam (Descenso Global)")
@time  res = solve(prob, OptimizationOptimisers.Adam(0.001); maxiters = 1500, 
            callback = callback)

println(">>> Fase 2: LBFGS (Refinamiento de Balance)")
prob2 = OptimizationProblem(optf, res.u)
@time res = solve(prob2, LBFGS(linesearch = BackTracking()); maxiters = 800)

println("Entrenamiento completado.")

# --- 7. POST-PROCESAMIENTO Y DIAGNÓSTICO (Versión Manual) ---

function post_balance_manual(θ)
    DeltaT = Th_in - Tc_in
    
    # Temperaturas físicas (°C)
    T_h_phys(x, e) = Th_model(x, e, θ) * DeltaT + Tc_in
    T_c_phys(x, e) = Tc_model(x, e, θ) * DeltaT + Tc_in

    # Configuración de integración
    detas = range(0.0, 1.0, length=200)
    de = 1.0 / (length(detas) - 1)
    factor = rho * cp * um * h_phys

    # Canal Caliente: In(x=0) -> Out(x=1)
    e_h_in  = sum(u_vel(e) * T_h_phys(0.0, e) for e in detas) * de
    e_h_out = sum(u_vel(e) * T_h_phys(1.0, e) for e in detas) * de
    Qh = (e_h_in - e_h_out) * factor
    
    # Canal Frío: In(x=1) -> Out(x=0)
    e_c_in  = sum(u_vel(abs(e)) * T_c_phys(1.0, -e) for e in detas) * de
    e_c_out = sum(u_vel(abs(e)) * T_c_phys(0.0, -e) for e in detas) * de
    Qc = (e_c_out - e_c_in) * factor

    println("\n" * "="^40)
    @printf("Q Cedido (Hot):  %10.2f W/m\n", Qh)
    @printf("Q Ganado (Cold): %10.2f W/m\n", Qc)
    @printf("Error Balance:    %10.4f %%\n", abs(Qh - Qc)/max(abs(Qh), 1e-9) * 100)
    println("="^40)
end

function visualizacion_manual(θ)
    xs = range(0.0, 1.0, length=50)
    etas_h = range(0.0, 1.0, length=50)
    etas_c = range(-1.0, 0.0, length=50)

    th_map = [Th_model(x, e, θ) for e in etas_h, x in xs]
    tc_map = [Tc_model(x, e, θ) for e in etas_c, x in xs]

    p1 = heatmap(xs, etas_h, th_map, title="Canal Caliente", cmap=:thermal)
    p2 = heatmap(xs, etas_c, tc_map, title="Canal Frío", cmap=:thermal)
    
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
    p1 = plot(xs, [Th_int Tc_int], label=["Hot" "Cold"], title="T en Interfaz (eta=0)", lw=2)

    # 2. Continuidad de Flujo (Derivada automática)
    qh = [kh * ForwardDiff.derivative(e -> Th_model(x, e, θ), 0.0) for x in xs]
    qc = [kc * ForwardDiff.derivative(e -> Tc_model(x, e, θ), 0.0) for x in xs]
    p2 = plot(xs, [qh qc], label=["Flux Hot" "Flux Cold"], title="Flujo en Interfaz", lw=2, ls=[:solid :dash])

    display(plot(p1, p2, layout=(2,1), size=(800, 600)))
end

# Ejecución
post_balance_manual(res.u)
visualizacion_manual(res.u)
diagnostico_interfaz_manual(res.u)