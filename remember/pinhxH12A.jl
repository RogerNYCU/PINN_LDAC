
# --- REPORTE FINAL DE BALANCE ---
# Q Cedido: 4.1674 W | Q Ganado: 20.4891 W
# Error Balance: 391.6520 %
# Codigo actual
using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots, Printf, Statistics, ComponentArrays, Random, ForwardDiff
using Base.Threads, LineSearches
using LineSearches
nthreads()
# --- 1. CONFIGURACIÓN FÍSICA ---
const Lx, h_phys = 5.0, 0.01 
const kh, kc = 1.0, 1.1
const rho, cp, um_final = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0
const DT_phys = Th_in - Tc_in
const xi_min, xi_max = -h_phys / kc, h_phys / kh

# --- 2. MALLA Y PRE-CÁLCULOS ---
const Nx, Nxi = 40, 30
const xs_train = collect(Float32, range(0.0, 1.0, length=Nx))
const xis_train = collect(Float32, range(xi_min, xi_max, length=Nxi))
const puntos = collect(Iterators.product(xs_train, xis_train))
const n_puntos = length(puntos)

function calc_u_eff(xi_v, um_val)
    eta = xi_v > 0 ? (xi_v * kh / h_phys) : (abs(xi_v) * kc / h_phys)
    vel = um_val * 1.5 * (1.0 - eta^2)
    return xi_v > 0 ? vel : -vel
end
const u_puntos = [Float32(calc_u_eff(p[2], 1.0)) for p in puntos]

# --- 3. RED NEURONAL ---
Random.seed!(42)
chain = Lux.Chain(Lux.Dense(2, 32, Lux.tanh), Lux.Dense(32, 32, Lux.tanh), Lux.Dense(32, 1))
ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

# --- 4. FUNCIÓN DE PÉRDIDA ROBUSTA Y PARALELA ---
function loss_function(p, params)
    um_val = params[1]
    nt = nthreads()
    
    # Definición global interna para que sea accesible en toda la pérdida
    predict(x_val, xi_val) = chain([x_val, xi_val], p, st)[1][1]
    
    l_pde_thread = zeros(eltype(p), nt + 4) 
    
    # 4.1 PDE Residual con Paralelismo
    Threads.@threads for i in 1:n_puntos
        idx = min(threadid(), length(l_pde_thread))
        x, xi = puntos[i]
        u_eff = u_puntos[i] * um_val 
        
        # Derivadas con ForwardDiff
        Tx = ForwardDiff.derivative(vx -> predict(vx, xi), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> predict(x, vvxi), vxi), xi)
        
        res = (rho * cp * u_eff * Tx) - (1.0f0/Lx * Txi_xi)
        
        # Causalidad agresiva para forzar la inclinación térmica
        w_c = xi > 0 ? exp(-4.0f0 * x) : exp(-4.0f0 * (1.0f0 - x))
        l_pde_thread[idx] += abs2(res / 1000.0f0) * w_c
    end
    
    loss_pde = sum(l_pde_thread) / n_puntos

    # 4.2 HBC (Interfaz xi=0) y Entradas
    l_bridge = sum(x -> abs2(predict(x, 0.0001f0) - predict(x, -0.0001f0)), xs_train) / Nx
    l_in_h = sum(xi -> abs2(predict(0.0f0, xi) - 1.0f0), filter(xi->xi>0, xis_train)) / (Nxi/2)
    l_in_c = sum(xi -> abs2(predict(1.0f0, xi) - 0.0f0), filter(xi->xi<0, xis_train)) / (Nxi/2)
    
    # 4.3 Paredes Adiabáticas
    l_wall = sum(x -> abs2(ForwardDiff.derivative(vxi -> predict(x, vxi), xi_min)) + 
                      abs2(ForwardDiff.derivative(vxi -> predict(x, vxi), xi_max)), xs_train) / Nx

    # 4.4 Balance de Energía (Integral)
    xis_h = filter(xi -> xi > 0, xis_train)
    xis_c = filter(xi -> xi < 0, xis_train)
    Qh_int = sum(xi -> calc_u_eff(xi, um_val) * (predict(0.0f0, xi) - predict(1.0f0, xi)), xis_h)
    Qc_int = sum(xi -> calc_u_eff(xi, um_val) * (predict(0.0f0, xi) - predict(1.0f0, xi)), xis_c)
    l_balance = abs2(Qh_int + Qc_int)

    return 2000.0f0 * loss_pde + 5000.0f0 * (l_in_h + l_in_c) + 2000.0f0 * l_bridge + 500.0f0 * l_wall + 1000.0f0 * l_balance
end

# --- 5. ENTRENAMIENTO MULTI-ETAPA ---
callback = function (p,l)
    if !@isdefined(iter_count) global iter_count =0 end
    global iter_count +=1
    if iter_count % 100 == 0 println("iteration: $iter_count | loss: $l") end
    return false
end
println(">>> Hilos configurados: ", nthreads())
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())

# Etapa 1: Adam (Fijar gradientes iniciales)
println("Fase 1: Adam Exploratorio...")
prob1 = OptimizationProblem(optf, θ_flat, [um_final/10.0f0])
res = solve(prob1, OptimizationOptimisers.Adam(0.002), maxiters=1500, callback=callback)

# Etapa 2: Adam (Velocidad completa)
println("Fase 2: Adam Convectivo...")
prob2 = OptimizationProblem(optf, res.u, [um_final])
res = solve(prob2, OptimizationOptimisers.Adam(0.001), callback=callback, maxiters=2000)

# Etapa 3: L-BFGS (Pulido de precisión)
println("Fase 3: L-BFGS...")
prob3 = OptimizationProblem(optf, res.u, [um_final])

# LBFGS simple sin especificar linesearch para evitar errores de librerías
res = solve(prob3, LBFGS() , callback=callback, maxiters=1000)

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

