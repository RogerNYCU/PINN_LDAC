# patron de Contraflujo perfecto pero es soft 

# Q Cedido: 41044.6911 W | Q Ganado: 38409.0470 W
# Error Balance: 6.4214 %
using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots, Printf, Statistics, ComponentArrays, Random, ForwardDiff
using Base.Threads
nthreads()
# --- 1. CONFIGURACIÓN FÍSICA ---
const Lx, h_phys = 5.0, 0.01 
const kh, kc = 1.0, 1.1
const rho, cp, um_final = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0
const DT_phys = Th_in - Tc_in
const xi_min, xi_max = -h_phys / kc, h_phys / kh

# --- 2. MALLA Y PRE-CÁLCULOS ---
const Nx, Nxi = 45, 35
const xs_train = collect(Float32, range(0.0, 1.0, length=Nx))
const xis_train = collect(Float32, range(xi_min, xi_max, length=Nxi))
const puntos = collect(Iterators.product(xs_train, xis_train))
const n_puntos = length(puntos)

function calc_u_eff(xi_v, um_val)
    # Perfil parabólico en espacio xi
    eta = xi_v > 0 ? (xi_v * kh / h_phys) : (abs(xi_v) * kc / h_phys)
    vel = um_val * 1.5 * (1.0 - eta^2)
    return xi_v > 0 ? vel : -vel
end
const u_puntos = [Float32(calc_u_eff(p[2], 1.0)) for p in puntos]

# --- 3. RED NEURONAL (Activación sin para mejores gradientes) ---
Random.seed!(1234)
chain = Lux.Chain(Lux.Dense(2, 40, Lux.sin), Lux.Dense(40, 40, Lux.sin), Lux.Dense(40, 1))
ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

# --- 4. FUNCIÓN DE PÉRDIDA ---
function loss_function(p, params)
    um_val = params[1]
    nt = nthreads()
    predict(x_v, xi_v) = chain([x_v, xi_v], p, st)[1][1]
    
    l_pde_thread = zeros(eltype(p), nt + 4) 
    
    # 4.1 Residual de la PDE paralelo
    Threads.@threads for i in 1:n_puntos
        idx = min(threadid(), length(l_pde_thread))
        x, xi = puntos[i]
        u_eff = u_puntos[i] * um_val 
        
        Tx = ForwardDiff.derivative(vx -> predict(vx, xi), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> predict(x, vvxi), vxi), xi)
        
        # Residuos normalizados por 1e5 para estabilidad numérica
        res = (rho * cp * u_eff * Tx) - (1.0f0/Lx * Txi_xi)
        w_c = xi > 0 ? exp(-3.0f0 * x) : exp(-3.0f0 * (1.0f0 - x))
        l_pde_thread[idx] += abs2(res / 100000.0f0) * w_c
    end
    
    loss_pde = sum(l_pde_thread) / n_puntos

    # 4.2 HBC y Condiciones de Contorno
    l_bridge = sum(x -> abs2(predict(x, 0.0001f0) - predict(x, -0.0001f0)), xs_train) / Nx
    l_in_h = sum(xi -> abs2(predict(0.0f0, xi) - 1.0f0), filter(xi->xi>0, xis_train)) / (Nxi/2)
    l_in_c = sum(xi -> abs2(predict(1.0f0, xi) - 0.0f0), filter(xi->xi<0, xis_train)) / (Nxi/2)
    
    # 4.3 Balance Global Directo
    xis_h = filter(xi -> xi > 0, xis_train)
    xis_c = filter(xi -> xi < 0, xis_train)
    Qh_int = sum(xi -> calc_u_eff(xi, um_val) * (predict(0.0f0, xi) - predict(1.0f0, xi)), xis_h)
    Qc_int = sum(xi -> calc_u_eff(xi, um_val) * (predict(0.0f0, xi) - predict(1.0f0, xi)), xis_c)
    l_balance = abs2(Qh_int + Qc_int)

    return 1000.0f0 * loss_pde + 8000.0f0 * (l_in_h + l_in_c) + 3000.0f0 * l_bridge + 1000.0f0 * l_balance
end

# --- 5. ENTRENAMIENTO ---
callback = function (p,l)
    if !@isdefined(iter_count) global iter_count =0 end
    global iter_count +=1
    if iter_count % 100 == 0 println("iteration: $iter_count | loss: $l") end
    return false
end
println(">>> Hilos activos: ", nthreads())
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())

println("Fase 1: Adam Exploratorio (Velocidad reducida)...")
@time res = solve(OptimizationProblem(optf, θ_flat, [0.005f0]), OptimizationOptimisers.Adam(0.005), maxiters=1500, callback=callback)

println("Fase 2: Adam Convectivo (Velocidad Nominal)...")
@time res = solve(OptimizationProblem(optf, res.u, [um_final]), OptimizationOptimisers.Adam(0.001), maxiters=2500, callback=callback)

println("Fase 3: L-BFGS (Refinamiento Final)...")
@time res_final = solve(OptimizationProblem(optf, res.u, [um_final]), LBFGS(), maxiters=1000, callback=callback)

# --- 6. REPORTE Y VISUALIZACIÓN ---
function final_report(p)
    f_p(x, xi) = chain([x, xi], p, st)[1][1]
    xs, xis = range(0.0, 1.0, length=100), range(xi_min, xi_max, length=100)
    
    # Balance de Energía real
    dxi_unit = (xi_max - xi_min) / 100
    Qh = sum(xi -> calc_u_eff(xi, um_final) * (f_p(0.0, xi) - f_p(1.0, xi)), range(1e-5, xi_max, length=100)) * rho * cp * DT_phys * dxi_unit
    Qc = sum(xi -> calc_u_eff(xi, um_final) * (f_p(1.0, xi) - f_p(0.0, xi)), range(xi_min, -1e-5, length=100)) * rho * cp * DT_phys * dxi_unit
    error_balance = round(abs(Qh - Qc)/max(abs(Qh), 1.0) * 100, digits=2)
println("\n--- FINAL REPORT ---")
println("Heat Transferred: $(round(Qh, digits=4)) W | Heat Gained: $(round(Qc, digits=4)) W")
println("Energy Balance Error: $(error_balance) % \n")

z = [f_p(x, xi) * DT_phys + Tc_in for xi in xis, x in xs]
p1 = heatmap(xs, xis, z, title="PINN Heatmap - Energy Balance: $(error_balance) %, Qh: $(round(Qh, digits=2)) W", 
             cmap=:turbo, xlabel="x (Length)", ylabel="xi (Thermal)")
hline!([0], color=:white, linestyle=:dash, label="Interface")
    display(p1)
end

final_report(res_final.u)