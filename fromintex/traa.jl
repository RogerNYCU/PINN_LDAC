using Lux, Optimization, OptimizationOptimJL, OptimizationOptimisers
using Plots, Statistics, ComponentArrays, Random, Zygote, ForwardDiff
using Base.Threads
using BenchmarkTools

# --- 1. CONFIGURACIÓN FÍSICA ---
const Lx, h_phys = 5.0, 0.01
const kh, kc = 1.0, 1.1
const rho, cp, um_final = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0
const DT_phys = Th_in - Tc_in
const xi_min, xi_max = Float32(-h_phys / kc), Float32(h_phys / kh)

# --- 2. MALLA REDUCIDA (Aceleración) ---
const Nx, Nxi = 40, 30
const xs_train = collect(Float32, range(0.0f0, 1.0f0, length=Nx))
const xis_train = collect(Float32, range(xi_min, xi_max, length=Nxi))
const puntos = collect(Iterators.product(xs_train, xis_train))
const n_puntos = length(puntos)

function calc_u_eff(xi_v, um_val)
    eta = xi_v > 0 ? (xi_v * kh / h_phys) : (abs(xi_v) * kc / h_phys)
    vel = um_val * 1.5 * (1.0 - eta^2)
    return xi_v > 0 ? vel : -vel
end

const u_puntos = [Float32(calc_u_eff(p[2], 1.0)) for p in puntos]

# --- 3. RED Y ANSATZ (Neumann-HardBC) ---
Random.seed!(1234)
chain = Lux.Chain(Lux.Dense(2, 32, Lux.sin), Lux.Dense(32, 32, Lux.sin), Lux.Dense(32, 1))
ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

function ansatz(x, xi, p)
    nn_out = chain([x, xi], p, st)[1][1]
    G = xi > 0 ? (1.0f0 - 0.4f0*x) : (0.4f0 - 0.4f0*x)
    dist_in = xi > 0 ? x : (1.0f0 - x)
    wall_mask = (xi - xi_min) * (xi_max - xi)
    return G + dist_in * wall_mask * nn_out
end

# --- 4. FUNCIÓN DE PÉRDIDA ESTABILIZADA ---
function loss_function(p, params)
    um_val = params[1]
    nt = 12 
    l_pde_v = zeros(eltype(p), nt)
    l_range_v = zeros(eltype(p), nt)
    puntos_por_hilo = partition(1:n_puntos, ceil(Int, n_puntos/nt))
    chunks = collect(puntos_por_hilo)
    Threads.@threads for tid in 1:nt
        if tid <= length(chunks)
            for i in chunks[tid]
                x, xi = puntos[i]
                u_eff = u_puntos[i] * um_val
                T_val = ansatz(x, xi, p)
                Tx = ForwardDiff.derivative(vx -> ansatz(vx, xi, p), x)
                Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> ansatz(x, vvxi, p), vxi), xi)
                res = (rho * cp * u_eff * Tx) - (1.0f0/Lx * Txi_xi)
                w_c = xi > 0 ? exp(-3.0f0 * x) : exp(-3.0f0 * (1.0f0 - x))
                l_pde_v[tid] += abs2(res / 100000.0f0) * w_c
                if T_val > 1.05f0
                    l_range_v[tid] += abs2(T_val - 1.05f0) * 100.0f0
                elseif T_val < -0.05f0
                    l_range_v[tid] += abs2(T_val + 0.05f0) * 100.0f0
                end
            end
        end
    end
    l_int = 0.0f0
    for x in xs_train
        T_gap = abs2(ansatz(x, 0.0001f0, p) - ansatz(x, -0.0001f0, p))
        l_int += 2000.0f0 * T_gap
    end
    return 1000.0f0 * (sum(l_pde_v)/n_puntos) + (l_int/Nx) + (sum(l_range_v)/n_puntos)
end

function partition(range, n)
    len = length(range)
    return [range[i:min(i + n - 1, len)] for i in 1:n:len]
end

callback = function (p, l)
    if !@isdefined(iter_count)
        global iter_count = 0
    end
    global iter_count += 1
    if iter_count % 100 == 0
        println("Iteración: $iter_count | Pérdida Total: $l")
    end
    return false
end

# --- 5. CONFIGURACIÓN DEL SOLVER ---
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, θ_flat, [um_final])

println(">>> INICIANDO ENTRENAMIENTO TURBO (MALLA REDUCIDA) <<<")
@time res = solve(prob, OptimizationOptimisers.Adam(0.005), callback=callback, maxiters=1000)

println("Refinamiento L-BFGS...")
@time res_final = solve(OptimizationProblem(optf, res.u, [um_final]), LBFGS(),callback=callback, maxiters=500)

# --- 6. ENERGY BALANCE AND VISUALIZATION ---
function report_and_graphics(p_final)
    get_T(x, xi) = ansatz(x, xi, p_final)
    nx, nxi = 100, 80
    xs = range(0.0, 1.0, length=nx)
    xis = range(xi_min, xi_max, length=nxi)
    dxi = (xi_max - xi_min) / (nxi - 1)
    xis_h = filter(xi -> xi > 0, xis)
    xis_c = filter(xi -> xi < 0, xis)
    Qh = sum(xis_h) do xi
        u = calc_u_eff(xi, um_final)
        u * (get_T(0.0, xi) - get_T(1.0, xi))
    end * rho * cp * DT_phys * dxi
    Qc = sum(xis_c) do xi
        u = calc_u_eff(xi, um_final)
        u * (get_T(0.0, xi) - get_T(1.0, xi))
    end * rho * cp * DT_phys * dxi
    println("========================================")
    println("   ENERGY BALANCE (HardBC)")
    println("========================================")
    println("Q Released (Hot Side):  ", Qh, " W")
    println("Q Gained (Cold Side):   ", abs(Qc), " W")
    println("Balance Error:          ", abs(Qh + Qc) / max(Qh, 1.0) * 100, " %")
    println("----------------------------------------")
    matriz_T = [get_T(x, xi) * DT_phys + Tc_in for xi in xis, x in xs]
    p1 = heatmap(xs, xis, matriz_T,
                 title="Temperature Heatmap (HardBC)",
                 xlabel="Position x",
                 ylabel="Coordinate xi",
                 cmap=:turbo,
                 colorbar_title=" Temp [°C]")
    hline!([0], color=:white, linestyle=:dash, label="Interface")
    hline!([xi_min, xi_max], color=:black, lw=2, label="Walls")
    T_h_interfaz = [get_T(x, 0.0001) * DT_phys + Tc_in for x in xs]
    T_c_interfaz = [get_T(x, -0.0001) * DT_phys + Tc_in for x in xs]
    p2 = plot(xs, T_h_interfaz, label="Hot Side (xi->0+)", color=:red, lw=2)
    plot!(xs, T_c_interfaz, label="Cold Side (xi->0-)", color=:blue, linestyle=:dash, lw=2)
    title!("Temperature Profile at the Interface")
    xlabel!("Axial Position x")
    ylabel!("Temperature [°C]")
    plot_final = plot(p1, p2, layout=(2,1), size=(800,800))
    display(plot_final)
end

report_and_graphics(res_final.u)
