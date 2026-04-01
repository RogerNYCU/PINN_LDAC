
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets
using BenchmarkTools
using OptimizationOptimisers, Plots, Printf, Statistics, ComponentArrays, Random, ForwardDiff
using Base.Threads

# ==========================================
# 1. PARÁMETROS Y GEOMETRÍA
# ==========================================
const Lx, h_phys = 5.0, 0.01 
const kh, kc = 1.0, 1.1
const rho, cp, um = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0

const xi_min, xi_max = -h_phys / kc, h_phys / kh
const DT_phys = Th_in - Tc_in

# ==========================================
# 2. RED Y ANSATZ
# ==========================================
Random.seed!(123)
chain = Lux.Chain(Lux.Dense(2, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin), Lux.Dense(32, 1))
ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

step_smooth(xi) = 0.5 * (1 + tanh(100.0 * xi))

function hbc_ansatz(x, xi, p)
    nn = chain([x, xi], p, st)[1][1]
    s = step_smooth(xi)
    t_in = s * 1.0 + (1.0 - s) * 0.0
    dist = s * x + (1.0 - s) * (x - 1.0)
    return t_in + dist * nn
end

# ==========================================
# 3. PDE PARALELIZADA (AQUÍ USAMOS LOS 12 CORES)
# ==========================================
const xs_train = collect(range(0.0, 1.0, length=35))
const xis_train = collect(range(xi_min, xi_max, length=25))
const puntos = collect(Iterators.product(xs_train, xis_train))

function u_vel_xi(xi_v)
    eta = xi_v > 0 ? (xi_v * kh / h_phys) : (abs(xi_v) * kc / h_phys)
    return um * 1.5 * (1.0 - eta^2)
end

function loss_function(p, _)
    # Determinamos el máximo de hilos disponibles para evitar BoundsError
    nt = nthreads()
    # Creamos un buffer compatible con los números Duales de ForwardDiff
    l_pde_hilos = [zeros(eltype(p), 1) for _ in 1:nt]
    
    # Usamos @threads para repartir la carga
    Threads.@threads for i in 1:length(puntos)
        # Obtenemos un ID seguro que nunca exceda nt
        tid = threadid()
        if tid > nt
            tid = nt # Salvaguarda para índices inesperados
        end

        x, xi = puntos[i]
        
        # Derivadas automáticas
        Tx = ForwardDiff.derivative(vx -> hbc_ansatz(vx, xi, p), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> hbc_ansatz(x, vvxi, p), vxi), xi)
        
        u_eff = xi > 0 ? u_vel_xi(xi) : -u_vel_xi(xi)
        
        adveccion = (rho * cp) * u_eff * Tx
        difusion  = (1.0 / Lx) * Txi_xi
        
        # PENALIZACIÓN PARA CONTRAFLUJO:
        # 1. Residuo de la PDE escalado.
        # 2. Penalizamos Tx ≈ 0 para forzar que la temperatura cambie en X.
        res_local = 800.0 * abs2(adveccion - difusion) + 300.0 * abs2(Tx)
        
        l_pde_hilos[tid][1] += res_local
    end
    
    # Reducción de los resultados de todos los hilos
    l_pde = sum(x -> x[1], l_pde_hilos) / length(puntos)

    # Pérdidas de contorno y acoplamiento (se calculan rápido en el hilo principal)
    l_wall = sum(x -> abs2(ForwardDiff.derivative(vxi -> hbc_ansatz(x, vxi, p), xi_min)) + 
                      abs2(ForwardDiff.derivative(vxi -> hbc_ansatz(x, vxi, p), xi_max)), xs_train) / length(xs_train)

    l_bridge = sum(x -> abs2(hbc_ansatz(x, 0.0005, p) - hbc_ansatz(x, -0.0005, p)), xs_train) / length(xs_train)

    return l_pde + 50.0 * l_wall + 1000.0 * l_bridge
end

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
println(">>> Usando $(nthreads()) hilos para el entrenamiento...")
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, θ_flat)

# Adam suele ser lento en paralelo, pero LBFGS vuela
println("Fase 1: Adam")
@time res = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 1200)

println("Fase 2: LBFGS (Aquí notarás la velocidad de los 12 núcleos)")
prob2 = OptimizationProblem(optf, res.u)
@time res = solve(prob2, LBFGS(); maxiters = 600)

# ==========================================
# 5. REPORTE
# ==========================================
function final_report(p)
    # Reutilizamos tu lógica de reporte
    xis_h, xis_c = range(1e-5, xi_max, length=100), range(xi_min, -1e-5, length=100)
    dxi_h, dxi_c = xi_max/100, abs(xi_min)/100
    f = rho * cp * DT_phys
    Qh = (sum(xi->u_vel_xi(xi)*hbc_ansatz(0.0,xi,p)*dxi_h, xis_h) - sum(xi->u_vel_xi(xi)*hbc_ansatz(1.0,xi,p)*dxi_h, xis_h)) * f
    Qc = (sum(xi->u_vel_xi(xi)*hbc_ansatz(0.0,xi,p)*dxi_c, xis_c) - sum(xi->u_vel_xi(xi)*hbc_ansatz(1.0,xi,p)*dxi_c, xis_c)) * f
    @printf("\nBalance Error: %.4f %%\n", abs(Qh - Qc)/max(abs(Qh), 1.0) * 100)
    println("heat Hot flow", Qh)
    println("heat Cold flow", Qc)
    xs_p, xis_p = range(0.0, 1.0, length=100), range(xi_min, xi_max, length=100)
    z = [hbc_ansatz(x, xi, p) * DT_phys + Tc_in for xi in xis_p, x in xs_p]
    heatmap(xs_p, xis_p, z, title="Heatmap - Multithreaded", cmap=:turbo)


end

final_report(res.u)