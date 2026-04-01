#en este codigo se implementa "balance de energia" en la interfaz para mejorar la exactitud, no funciono




using Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets
using OptimizationOptimisers, Plots, Printf, Statistics, ComponentArrays, Random, ForwardDiff
using Base.Threads

# ==========================================
# 1. PARÁMETROS FÍSICOS
# ==========================================
const Lx, h_phys = 5.0, 0.01 
const kh, kc = 1.0, 1.1
const rho, cp, um = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0
const DT_phys = Th_in - Tc_in

# Coordenada térmica xi (HBC implica escalar por conductividad)
const xi_min, xi_max = -h_phys / kc, h_phys / kh

# ==========================================
# 2. RED Y ANSATZ
# ==========================================
Random.seed!(123)
chain = Lux.Chain(Lux.Dense(2, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin), Lux.Dense(32, 1))
ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

# Predicción adimensional
function get_T(x, xi, p)
    return chain([x, xi], p, st)[1][1]
end

function u_vel_xi(xi_v)
    # Perfil parabólico adaptado a la coordenada xi
    eta = xi_v > 0 ? (xi_v * kh / h_phys) : (abs(xi_v) * kc / h_phys)
    return um * 1.5 * (1.0 - eta^2)
end

# ==========================================
# 3. PUNTOS DE ENTRENAMIENTO
# ==========================================
const xs_train = collect(range(0.0, 1.0, length=35))
const xis_train = collect(range(xi_min, xi_max, length=25))
const puntos = collect(Iterators.product(xs_train, xis_train))

# ==========================================
# 4. FUNCIÓN DE PÉRDIDA (CORREGIDA PARA HILOS)
# ==========================================
function loss_function(p, _)
    # Buffer de hilos robusto
    n_total_threads = nthreads()
    l_pde_hilos = [zeros(eltype(p), 1) for _ in 1:n_total_threads]
    
    Threads.@threads for i in 1:length(puntos)
        tid = threadid()
        # Salvaguarda para el ID del hilo
        idx = min(tid, n_total_threads)
        
        x, xi = puntos[i]
        
        # Derivadas (HBC: xi simplifica la difusión a T_xi_xi)
        Tx = ForwardDiff.derivative(vx -> get_T(vx, xi, p), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> get_T(x, vvxi, p), vxi), xi)
        
        u_eff = xi > 0 ? u_vel_xi(xi) : -u_vel_xi(xi)
        
        adveccion = (rho * cp) * u_eff * Tx
        difusion  = (1.0 / Lx) * Txi_xi
        
        # Peso de Causalidad: Forzamos que aprenda la entrada primero
        w_causal = xi > 0 ? exp(-3.0 * x) : exp(-3.0 * (1.0 - x))
        
        l_pde_hilos[idx][1] += (adveccion - difusion)^2 * w_causal
    end
    
    loss_pde = sum(v -> v[1], l_pde_hilos) / length(puntos)

    # HBC: Puente de continuidad en xi=0
    # Usamos un pequeño epsilon para evitar la discontinuidad numérica en la red
    l_bridge = sum(x -> abs2(get_T(x, 0.0001, p) - get_T(x, -0.0001, p)), xs_train) / length(xs_train)

    # Entradas (Soft Constraints para permitir gradientes)
    l_in_h = sum(xi -> abs2(get_T(0.0, xi, p) - 1.0), filter(xi->xi>0, xis_train))
    l_in_c = sum(xi -> abs2(get_T(1.0, xi, p) - 0.0), filter(xi->xi<0, xis_train))

    # Paredes adiabáticas
    l_wall = sum(x -> abs2(ForwardDiff.derivative(vxi -> get_T(x, vxi, p), xi_min)) + 
                      abs2(ForwardDiff.derivative(vxi -> get_T(x, vxi, p), xi_max)), xs_train)

    return 1000.0 * loss_pde + 1000.0 * l_bridge + 500.0 * (l_in_h + l_in_c) + 100.0 * l_wall
end
callback = function (p,l)
    if !@isdefined(iter_count) global iter_count =0 end
    global iter_count +=1
    if iter_count % 100 == 0 println("iteration: $iter_count | loss: $l") end
    return false
end
# ==========================================
# 5. ENTRENAMIENTO
# ==========================================
println(">>> Entrenando con $(nthreads()) hilos...")
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, θ_flat)

# Adam largo para establecer la pendiente térmica
println("Fase 1: Adam")
@time res = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 2500)

# LBFGS para convergencia de alta precisión
println("Fase 2: LBFGS")
prob2 = OptimizationProblem(optf, res.u)
@time res = solve(prob2, LBFGS(); maxiters = 800)

# ==========================================
# 6. REPORTE Y BALANCE
# ==========================================
function final_report(p)
    xis_h, xis_c = range(1e-5, xi_max, length=100), range(xi_min, -1e-5, length=100)
    dxi = (xi_max - xi_min) / 200
    
    # Energy balance integral
    Qh = sum(xi -> u_vel_xi(xi)*(get_T(0.0, xi, p) - get_T(1.0, xi, p)), xis_h) * rho * cp * DT_phys * dxi
    Qc = sum(xi -> u_vel_xi(xi)*(get_T(1.0, xi, p) - get_T(0.0, xi, p)), xis_c) * rho * cp * DT_phys * dxi

    println("\nEnergy Balance:")
    println("Heat Released: $(round(Qh, digits=4)) W | Heat Gained: $(round(Qc, digits=4)) W")
    println("Balance Error: $(round(abs(Qh - Qc)/max(Qh, 1.0) * 100, digits=4)) %")

    xs_p, xis_p = range(0.0, 1.0, length=100), range(xi_min, xi_max, length=100)
    z = [get_T(x, xi, p) * DT_phys + Tc_in for xi in xis_p, x in xs_p]
    heatmap(xs_p, xis_p, z, title="PINN: Counter-flow with HBC", cmap=:turbo, xlabel="x", ylabel="xi")


end

final_report(res.u)