
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets,
      OptimizationOptimisers, Plots, Printf, Statistics, Zygote, ComponentArrays
using Random, ForwardDiff, BenchmarkTools


# --- 1. PARÁMETROS FÍSICOS ---
Lx, h_phys = 2.0, 0.05
kh, kc = 1.0, 1.1
rho, cp, um = 1000.0, 4180.0, 0.05
Th_in, Tc_in = 80.0, 20.0

Ch = (kh * Lx) / (rho * cp * um * h_phys^2)
Cc = (kc * Lx) / (rho * cp * um * h_phys^2)

# --- 2. ARQUITECTURA ---
# Siren (sin) es excelente para segundas derivadas
chain_h = Lux.Chain(Lux.Dense(2, 50, Lux.sin), Lux.Dense(50, 50, Lux.sin), Lux.Dense(50, 1))
chain_c = Lux.Chain(Lux.Dense(2, 50, Lux.sin), Lux.Dense(50, 50, Lux.sin), Lux.Dense(50, 1))

ps_h, st_h = Lux.setup(Random.default_rng(), chain_h)
ps_c, st_c = Lux.setup(Random.default_rng(), chain_c)
θ_init = ComponentVector(h = ps_h, c = ps_c)

# --- 3. FUNCIONES DE ACCESO SEGURO ---
# Usamos ForwardDiff para derivadas internas respecto a x y eta
u_vel(e) = 1.5 * (1.0 - e^2)

# Th(x, e) con HBC: Th(0, e) = 1.0
Th_model(x, e, p) = 1.0 + x * chain_h([x, e], p.h, st_h)[1][1]
# Tc(x, e) con HBC: Tc(1, e) = 0.0
Tc_model(x, e, p) = 0.0 + (x - 1.0) * chain_c([x, e], p.c, st_c)[1][1]

# --- 4. CÁLCULO DE DERIVADAS (Evitando llvmcall) ---
function get_derivatives_h(x, e, p)
    # Primera derivada x (Advección)
    Tx = ForwardDiff.derivative(vx -> Th_model(vx, e, p), x)
    # Segunda derivada e (Difusión)
    Tee = ForwardDiff.derivative(ve -> ForwardDiff.derivative(vve -> Th_model(x, vve, p), ve), e)
    return Tx, Tee
end

function get_derivatives_c(x, e, p)
    Tx = ForwardDiff.derivative(vx -> Tc_model(vx, e, p), x)
    Tee = ForwardDiff.derivative(ve -> ForwardDiff.derivative(vve -> Tc_model(x, vve, p), ve), e)
    return Tx, Tee
end

# --- 5. FUNCIÓN DE PÉRDIDA ---
xs = range(0.0, 1.0, length=30)
es_h = range(0.0, 1.0, length=15)
es_c = range(-1.0, 0.0, length=15)

function total_loss(θ, p)
    # PDE Canal Caliente
    l_h = sum(Iterators.product(xs, es_h)) do (x, e)
        Tx, Tee = get_derivatives_h(x, e, θ)
        abs2(u_vel(e) * Tx - Ch * Tee)
    end / (30*15)

    # PDE Canal Frío
    l_c = sum(Iterators.product(xs, es_c)) do (x, e)
        Tx, Tee = get_derivatives_c(x, e, θ)
        abs2(u_vel(e) * Tx - Cc * Tee)
    end / (30*15)

    # INTERFAZ (e=0): Continuidad de T y Flujo
    l_int = sum(xs) do x
        t0_h = Th_model(x, 0.0, θ)
        t0_c = Tc_model(x, 0.0, θ)
        
        # dTh/de y dTc/de
        qh = kh * ForwardDiff.derivative(ve -> Th_model(x, ve, θ), 0.0)
        qc = kc * ForwardDiff.derivative(ve -> Tc_model(x, ve, θ), 0.0)
        
        abs2(t0_h - t0_c) + abs2(qh - qc)
    end / 30

    # PAREDES ADIABÁTICAS
    l_wall = sum(xs) do x
        abs2(ForwardDiff.derivative(ve -> Th_model(x, ve, θ), 1.0)) +
        abs2(ForwardDiff.derivative(ve -> Tc_model(x, ve, θ), -1.0))
    end / 30

    # Aumentamos el peso de 20.0 a 100.0 o más
    return l_h + l_c + 100.0 * l_int + l_wall
end

# --- 6. OPTIMIZACIÓN ---
optf = OptimizationFunction(total_loss, Optimization.AutoZygote())
prob = OptimizationProblem(optf, θ_init)

println("Iniciando Adam...")
res = solve(prob, OptimizationOptimisers.Adam(0.002); maxiters = 1000)

println("Iniciando LBFGS...")
prob2 = OptimizationProblem(optf, res.u)
res = solve(prob2, LBFGS(); maxiters = 500)

using Printf, Plots

function visualizar_y_balance_final(θ)
    # --- 1. GENERACIÓN DE DATOS PARA EL MAPA ---
    nx, neta = 100, 50
    xs_plot = range(0.0, 1.0, length=nx)
    etas_h = range(0.0, 1.0, length=neta)
    etas_c = range(-1.0, 0.0, length=neta)

    # Evaluación de las redes
    # Canal Caliente (Superior)
    Z_h = [Th_model(x, e, θ) * (Th_in - Tc_in) + Tc_in for e in etas_h, x in xs_plot]
    # Canal Frío (Inferior)
    Z_c = [Tc_model(x, e, θ) * (Th_in - Tc_in) + Tc_in for e in etas_c, x in xs_plot]

    # Unir dominios y coordenadas físicas
    Z_total = vcat(Z_c, Z_h)
    y_phys = vcat(etas_c .* h_phys, etas_h .* h_phys)
    x_phys = xs_plot .* Lx

    # --- 2. HEATMAP UNIFICADO ---
    p1 = heatmap(x_phys, y_phys, Z_total,
        aspect_ratio=:auto,
        seriescolor=:turbo,
        title="PINN Multi-Red: Perfil Térmico Final",
        xlabel="Distancia x (m)",
        ylabel="Altura y (m)",
        colorbar_title="Temperatura (°C)")
    
    # Línea divisoria de la interfaz
    hline!([0], color=:white, linestyle=:dash, label="Interfaz", lw=1.5)
    display(p1)

    # --- 3. CÁLCULO RIGUROSO DEL BALANCE ---
    # Usamos integración numérica de Simpson o Trapecio sobre los perfiles de salida
    n_int = 200
    de = 1.0 / n_int
    area_factor = rho * cp * um * h_phys # Conversión a Watts/m

    # Perfil de velocidad: 1.5 * um * (1 - e^2) -> normalizado es 1.5 * (1 - e^2)
    vel(e) = 1.5 * (1.0 - e^2)

    # Canal Caliente: Entrada x=0 (T=1.0), Salida x=1.0
    # Energía In = ∫ rho*cp*u*T_in dy
    e_hot_in = sum(e -> vel(e) * 1.0 * de, range(0, 1, length=n_int))
    # Energía Out = ∫ rho*cp*u*T_out(e) dy
    e_hot_out = sum(e -> vel(e) * Th_model(1.0, e, θ) * de, range(0, 1, length=n_int))
    
    Q_cedido = (e_hot_in - e_hot_out) * area_factor

    # Canal Frío: Entrada x=1.0 (T=0.0), Salida x=0.0
    # Energía In = 0 (T_in = 0)
    e_cold_in = 0.0
    # Energía Out = ∫ rho*cp*u*T_out(e) dy
    e_cold_out = sum(e -> vel(e) * Tc_model(0.0, e, θ) * de, range(-1, 0, length=n_int))
    
    Q_ganado = (e_cold_out - e_cold_in) * area_factor

    # --- 4. IMPRESIÓN DE RESULTADOS ---

    print("   REPORTE DE BALANCE TÉRMICO (PINN)\n")
    println("Q Cedido (Lado Caliente):  ", Q_cedido)
    println("Q Ganado (Lado Frío):     ", Q_ganado)
    println("-"^40 * "\n")
    println("Error de Conservación:    ", abs(Q_cedido - Q_ganado)/Q_cedido * 100)
    println("Diferencia vs FVM (~15kW): ", abs(Q_cedido - 14960.78))

    return p1
end

# Ejecutar después del entrenamiento
visualizar_y_balance_final(res.u)

# Graficar T en x=1.0 para todo el ancho y
y_range = vcat(range(-1,0,length=50), range(0,1,length=50))
temp_profile = [e < 0 ? Tc_model(1.0, e, res.u) : Th_model(1.0, e, res.u) for e in y_range]

plot(y_range, temp_profile, lw=2, title="Perfil Transversal en x=1.0",
     xlabel="eta (Posición vertical)", ylabel="Temperatura Normalizada", label="PINN")
vline!([0], color=:black, linestyle=:dash, label="Interfaz")