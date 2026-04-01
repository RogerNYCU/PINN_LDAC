#este es el primer intento con hbc en todos lados

using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots, Printf, Statistics, ComponentArrays, Random, ForwardDiff
using Base.Threads

# --- 1. CONFIGURACIÓN FÍSICA ---
const Lx, h_phys = 5.0, 0.01 
const kh, kc = 1.0, 1.1
const rho, cp, um_final = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0
const DT_phys = Th_in - Tc_in
const xi_min, xi_max = -h_phys / kc, h_phys / kh

# --- 2. MALLA REFINADA ---
const Nx, Nxi = 60, 50 
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
Random.seed!(1234)
chain = Lux.Chain(Lux.Dense(2, 40, Lux.sin), Lux.Dense(40, 40, Lux.sin), Lux.Dense(40, 1))
ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

# --- 4. FUNCIÓN DE PÉRDIDA PARALELIZADA (12 CORES) ---
# --- 4. FUNCIÓN DE PÉRDIDA PARALELIZADA (SEGURA) ---
function loss_function(p, params)
    um_val = params[1]
    nt = nthreads()
    predict(x_v, xi_v) = chain([x_v, xi_v], p, st)[1][1]
    
    # Añadimos un margen de seguridad al tamaño de los vectores (nt + 1)
    # y usamos clamp para asegurar que el índice nunca exceda el tamaño.
    l_pde_v = zeros(eltype(p), nt + 1)
    l_in_v  = zeros(eltype(p), nt + 1)
    l_int_v = zeros(eltype(p), nt + 1)
    l_wal_v = zeros(eltype(p), nt + 1)

    # 4.1 PDE PARALELA
    Threads.@threads for i in 1:n_puntos
        # clamp asegura que el índice esté entre 1 y nt+1
        tid = clamp(threadid(), 1, nt + 1)
        x, xi = puntos[i]
        u_eff = u_puntos[i] * um_val 
        
        Tx = ForwardDiff.derivative(vx -> predict(vx, xi), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> predict(x, vvxi), vxi), xi)
        
        res = (rho * cp * u_eff * Tx) - (1.0f0/Lx * Txi_xi)
        l_pde_v[tid] += abs2(res / 100000.0f0)
    end

    # 4.2 ENTRADAS PARALELAS
    Threads.@threads for i in 1:Nxi
        tid = clamp(threadid(), 1, nt + 1)
        xi = xis_train[i]
        if xi > 0 
            l_in_v[tid] += abs2(predict(0.0f0, xi) - 1.0f0)
        else
            l_in_v[tid] += abs2(predict(1.0f0, xi) - 0.0f0)
        end
    end

    # 4.3 INTERFAZ Y PAREDES PARALELAS
    Threads.@threads for i in 1:Nx
        tid = clamp(threadid(), 1, nt + 1)
        x = xs_train[i]
        
        # Interfaz
        T_gap = abs2(predict(x, 0.0001f0) - predict(x, -0.0001f0))
        grad_h = ForwardDiff.derivative(vxi -> predict(x, vxi), 0.0001f0)
        grad_c = ForwardDiff.derivative(vxi -> predict(x, vxi), -0.0001f0)
        l_int_v[tid] += T_gap + abs2(kh * grad_h - kc * grad_c)

        # Paredes
        grad_up = ForwardDiff.derivative(vxi -> predict(x, vxi), xi_max)
        grad_low = ForwardDiff.derivative(vxi -> predict(x, vxi), xi_min)
        l_wal_v[tid] += abs2(grad_up) + abs2(grad_low)
    end

    return 1000.0f0 * (sum(l_pde_v)/n_puntos) + 
           5000.0f0 * (sum(l_in_v)/Nxi) + 
           3000.0f0 * (sum(l_int_v)/Nx) + 
           2000.0f0 * (sum(l_wal_v)/Nx)
end

# --- 5. ENTRENAMIENTO ---
println(">>> EJECUTANDO EN 12 NÚCLEOS (MODO TURBO) <<<")
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())

println("Fase 1: Adam Exploratorio...")
@time res = solve(OptimizationProblem(optf, θ_flat, [0.005f0]), Adam(0.005), maxiters=1500)

println("Fase 2: Adam Convectivo...")
@time res = solve(OptimizationProblem(optf, res.u, [um_final]), Adam(0.001), maxiters=2500)

println("Fase 3: L-BFGS (Refinamiento)...")
@time res_final = solve(OptimizationProblem(optf, res.u, [um_final]), LBFGS(), maxiters=500)