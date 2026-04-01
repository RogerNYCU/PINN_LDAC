#este codigo usa 2 redes
# buenos gradientes, error 27%, Q 13.6 kW/m
#usa HBC para la entrada de los fluidos, pero es soft para la interfaz y las paredes adiabaticas


using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers, Plots, Printf, Zygote, ForwardDiff, ComponentArrays, Random, Statistics
using BenchmarkTools
# --- 1. CONFIGURACIÓN ---
Lx, h_phys = 2.0, 0.05
kh, kc = 1.0, 1.1
rho, cp, um = 1000.0, 4180.0, 0.05
Th_in, Tc_in = 80.0, 20.0

Ch = (kh * Lx) / (rho * cp * um * h_phys^2)
Cc = (kc * Lx) / (rho * cp * um * h_phys^2)

rng = Random.seed!(42)
chain_h = Lux.Chain(Lux.Dense(2, 25, Lux.sin), Lux.Dense(25, 25, Lux.sin), Lux.Dense(25, 1))
chain_c = Lux.Chain(Lux.Dense(2, 25, Lux.sin), Lux.Dense(25, 25, Lux.sin), Lux.Dense(25, 1))

ps_h, st_h = Lux.setup(rng, chain_h)
ps_c, st_c = Lux.setup(rng, chain_c)
θ_init = ComponentVector(h = ps_h, c = ps_c)

# --- 2. MODELOS CON HBC ---
# Th(x,e): x=0 => T=1.0 | Tc(x,e): x=1 => T=0.0
Th_func(x, e, p) = 1.0 + x * chain_h([x, e], p.h, st_h)[1][1]
Tc_func(x, e, p) = 0.0 + (x - 1.0) * chain_c([x, e], p.c, st_c)[1][1]

# --- 3. DERIVADAS CON FORWARDDIFF (Estables) ---
function pde_h(x, e, p)
    Tx = ForwardDiff.derivative(vx -> Th_func(vx, e, p), x)
    Te = ForwardDiff.derivative(ve -> Th_func(x, ve, p), e)
    Tee = ForwardDiff.derivative(ve -> ForwardDiff.derivative(vve -> Th_func(x, vve, p), ve), e)
    return (1.5 * (1.0 - e^2)) * Tx - Ch * Tee
end

function pde_c(x, e, p)
    Tx = ForwardDiff.derivative(vx -> Tc_func(vx, e, p), x)
    Te = ForwardDiff.derivative(ve -> Tc_func(x, ve, p), e)
    Tee = ForwardDiff.derivative(ve -> ForwardDiff.derivative(vve -> Tc_func(x, vve, p), ve), e)
    return (1.5 * (1.0 - e^2)) * Tx - Cc * Tee
end

# --- 4. FUNCIÓN DE PÉRDIDA VECTORIZADA ---
# Pre-generamos los puntos para evitar errores de scope
pts_x = collect(range(0.0, 1.0, length=25))
pts_eh = collect(range(0.0, 1.0, length=15))
pts_ec = collect(range(-1.0, 0.0, length=15))

function loss_fn(θ, _)
    # PDE Losses
    loss_h = sum(abs2, pde_h(x, e, θ) for x in pts_x, e in pts_eh) / (25*15)
    loss_c = sum(abs2, pde_c(x, e, θ) for x in pts_x, e in pts_ec) / (25*15)
    
    # Interface (e=0)
    l_int = sum(pts_x) do x
        t_h = Th_func(x, 0.0, θ)
        t_c = Tc_func(x, 0.0, θ)
        qh = kh * ForwardDiff.derivative(ve -> Th_func(x, ve, θ), 0.0)
        qc = kc * ForwardDiff.derivative(ve -> Tc_func(x, ve, θ), 0.0)
        abs2(t_h - t_c) + abs2(qh - qc)
    end / 25
    
    # Adiabatic walls (e=1, e=-1)
    l_wall = sum(pts_x) do x
        abs2(ForwardDiff.derivative(ve -> Th_func(x, ve, θ), 1.0)) +
        abs2(ForwardDiff.derivative(ve -> Tc_func(x, ve, θ), -1.0))
    end / 25
    
    return loss_h + loss_c + 20.0 * l_int + l_wall
end

# --- 5. ENTRENAMIENTO ---
callback = function (p, l)
    if !@isdefined(iter_count) global iter_count = 0 end
    global iter_count += 1
    if iter_count % 100 == 0 println("Iteración: $iter_count | Pérdida: $l") end
    return false
end
println("Entrenando...")
optf = OptimizationFunction(loss_fn, Optimization.AutoZygote())
prob = OptimizationProblem(optf, θ_init)

@time res = solve(prob, OptimizationOptimisers.Adam(0.002); maxiters = 2, callback = callback)
prob2 = OptimizationProblem(optf, res.u)
@time res = solve(prob2, LBFGS(); maxiters = 2, callback = callback)

# --- 6. BALANCE DE ENERGÍA ---
# Graficar T en x=1.0 para todo el ancho y

x_range = vcat(range(0.0, 1.0, length=50))
temp_ih= [Th_func(xi, 0.0, res.u) for xi in x_range]
temp_ic= [Tc_func(xi, 0.0, res.u) for xi in x_range]
plot(x_range, temp_ih, lw=2, label="Th(x,0)", title="Perfil a lo largo de x en la interfaz")
plot!(x_range, temp_ic, lw=2, label="Tc(x,0)")

using Printf, Plots

function visualizar_y_balance_final(θ)
    # --- 1. GENERACIÓN DE DATOS PARA EL MAPA ---
    nx, neta = 100, 50
    xs_plot = range(0.0, 1.0, length=nx)
    etas_h = range(0.0, 1.0, length=neta)
    etas_c = range(-1.0, 0.0, length=neta)

    # Evaluación de las redes
    # Canal Caliente (Superior)
    Z_h = [Th_func(x, e, θ) * (Th_in - Tc_in) + Tc_in for e in etas_h, x in xs_plot]
    # Canal Frío (Inferior)
    Z_c = [Tc_func(x, e, θ) * (Th_in - Tc_in) + Tc_in for e in etas_c, x in xs_plot]

    # Unir dominios y coordenadas físicas
    Z_total = vcat(Z_c, Z_h)
    y_phys = vcat(etas_c .* h_phys, etas_h .* h_phys)
    x_phys = xs_plot .* Lx



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
    e_hot_out = sum(e -> vel(e) * Th_func(1.0, e, θ) * de, range(0, 1, length=n_int))
    
    Q_cedido = (e_hot_in - e_hot_out) * area_factor

    # Canal Frío: Entrada x=1.0 (T=0.0), Salida x=0.0
    # Energía In = 0 (T_in = 0)
    e_cold_in = 0.0
    # Energía Out = ∫ rho*cp*u*T_out(e) dy
    e_cold_out = sum(e -> vel(e) * Tc_func(0.0, e, θ) * de, range(-1, 0, length=n_int))
    
    Q_ganado = (e_cold_out - e_cold_in) * area_factor
    error_balance = round(abs(Q_cedido - Q_ganado)/Q_cedido * 100; digits=4)
    # --- 4. IMPRESIÓN DE RESULTADOS ---
    println("\n" * "="^40)
    println("   REPORTE DE BALANCE TÉRMICO (PINN)")
    println("-"^40)
    println("Q Cedido (Lado Caliente):  $(round(Q_cedido; digits=2)) W/m")
    println("Q Ganado (Lado Frío):      $(round(Q_ganado; digits=2)) W/m")
    println("-"^40)
    println("Error de Conservación:     $error_balance %")
    println("Diferencia vs FVM (~15kW): $(round(abs(Q_cedido - 14960.78); digits=2)) W/m")
    println("="^40)
        # --- 2. UNIFIED HEATMAP ---
    p1 = heatmap(x_phys, y_phys, Z_total,
        aspect_ratio=:auto,
        seriescolor=:thermal,
        title=title="PINN HBC: Physical Temperature $error_balance %, $(round(Q_cedido, digits=2)) ",
        xlabel="Distance x (m)",
        ylabel="Height y (m)",
        colorbar_title="Temperature (°C)")
        
    
    # Línea divisoria de la interfaz
    hline!([0], color=:white, linestyle=:dash, label="Interfaz", lw=1.5)
    display(p1)
    savefig(p1, "remember/25Hx14.png")
    return p1
end

# Ejecutar después del entrenamiento
visualizar_y_balance_final(res.u)

