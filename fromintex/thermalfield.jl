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
function loss_function(p, params)
    um_val = params[1]
    nt = nthreads()
    predict(x_v, xi_v) = chain([x_v, xi_v], p, st)[1][1]
    
    # Vectores para reducir resultados por hilo (evita colisiones)
    l_pde_v = zeros(eltype(p), nt)
    l_in_v  = zeros(eltype(p), nt)
    l_int_v = zeros(eltype(p), nt)
    l_wal_v = zeros(eltype(p), nt)

    # 4.1 PDE PARALELA
    Threads.@threads for i in 1:n_puntos
        tid = threadid()
        x, xi = puntos[i]
        u_eff = u_puntos[i] * um_val 
        Tx = ForwardDiff.derivative(vx -> predict(vx, xi), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> predict(x, vvxi), vxi), xi)
        res = (rho * cp * u_eff * Tx) - (1.0f0/Lx * Txi_xi)
        l_pde_v[tid] += abs2(res / 100000.0f0)
    end

    # 4.2 ENTRADAS PARALELAS (HBC In)
    Threads.@threads for i in 1:Nxi
        tid = threadid()
        xi = xis_train[i]
        if xi > 0 # Hot in x=0
            l_in_v[tid] += abs2(predict(0.0f0, xi) - 1.0f0)
        else      # Cold in x=1
            l_in_v[tid] += abs2(predict(1.0f0, xi) - 0.0f0)
        end
    end

    # 4.3 INTERFAZ Y PAREDES PARALELAS (HBC Interface & Walls)
    Threads.@threads for i in 1:Nx
        tid = threadid()
        x = xs_train[i]
        
        # Interfaz xi=0
        T_gap = abs2(predict(x, 0.0001f0) - predict(x, -0.0001f0))
        grad_h = ForwardDiff.derivative(vxi -> predict(x, vxi), 0.0001f0)
        grad_c = ForwardDiff.derivative(vxi -> predict(x, vxi), -0.0001f0)
        l_int_v[tid] += T_gap + abs2(kh * grad_h - kc * grad_c)

        # Paredes (Adiabáticas)
        grad_up = ForwardDiff.derivative(vxi -> predict(x, vxi), xi_max)
        grad_low = ForwardDiff.derivative(vxi -> predict(x, vxi), xi_min)
        l_wal_v[tid] += abs2(grad_up) + abs2(grad_low)
    end

    return 1000.0f0 * (sum(l_pde_v)/n_puntos) + 
           5000.0f0 * (sum(l_in_v)/Nxi) + 
           3000.0f0 * (sum(l_int_v)/Nx) + 
           2000.0f0 * (sum(l_wal_v)/Nx)
end

# Función para monitorear el progreso
callback = function (p, l)
    # Usamos una variable global o de scope superior para contar iteraciones
    if !@isdefined(iter_count)
        global iter_count = 0
    end
    global iter_count += 1
    
    if iter_count % 1 == 0
        @printf("Iteración: %d | Pérdida Total: %.6e\n", iter_count, l)
    end
    return false # Retornar true detendría el entrenamiento prematuramente
end

# --- 5. ENTRENAMIENTO ---
println(">>> EJECUTANDO EN 12 NÚCLEOS (MODO TURBO) <<<")
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())

println("Fase 1: Adam Exploratorio...")
@time res = solve(OptimizationProblem(optf, θ_flat, [0.005f0]), OptimizationOptimisers.Adam(0.005), callback = callback, maxiters=1500)

println("Fase 2: Adam Convectivo...")
@time res = solve(OptimizationProblem(optf, res.u, [um_final]), OptimizationOptimisers.Adam(0.001), callback = callback, maxiters=2500)

println("Fase 3: L-BFGS (Refinamiento)...")
@time res_final = solve(OptimizationProblem(optf, res.u, [um_final]), LBFGS(), callback = callback, maxiters=2)

# --- 6. REPORTE FINAL Y BALANCE DE ENERGÍA ---
function generate_final_results(p_final)
    # Función de predicción local
    f_p(x, xi) = chain([x, xi], p_final, st)[1][1]
    
    # Malla de alta resolución para integración
    nx_eval, nxi_eval = 120, 100
    xs = range(0.0f0, 1.0f0, length=nx_eval)
    xis = range(xi_min, xi_max, length=nxi_eval)
    dxi = (xi_max - xi_min) / (nxi_eval - 1)

    # 1. Cálculo de Q (Watts) usando integración trapezoide sobre el flujo másico
    # Q = integral( rho * cp * u(xi) * (T_in - T_out) dxi )
    xis_h = filter(xi -> xi > 0, xis)
    xis_c = filter(xi -> xi < 0, xis)

    # Calor Cedido (Lado Caliente)
    Qh = sum(xis_h) do xi
        u = calc_u_eff(xi, um_final)
        # Delta T entre entrada (x=0) y salida (x=1)
        u * (f_p(0.0f0, xi) - f_p(1.0f0, xi))
    end * rho * cp * DT_phys * dxi

    # Calor Ganado (Lado Frío)
    Qc = sum(xis_c) do xi
        u = calc_u_eff(xi, um_final)
        # Delta T entre salida (x=0) y entrada (x=1) para contraflujo
        u * (f_p(0.0f0, xi) - f_p(1.0f0, xi))
    end * rho * cp * DT_phys * dxi

    # 2. Reporte en Consola
    println("\n" * "="^40)
    println("   BALANCE DE ENERGÍA (HBC TOTAL)")
    println("="^40)
    println("Q Cedido (Hot Side):  $(round(Qh, digits=4)) W")
    println("Q Ganado (Cold Side): $(round(abs(Qc), digits=4)) W")
    println("Error de Balance:     $(round(abs(Qh + Qc) / max(Qh, 1.0) * 100, digits=4)) %")
    println("-"^40)

    # 3. Heatmap de Temperatura
    z = [f_p(x, xi) * DT_phys + Tc_in for xi in xis, x in xs]
    p1 = heatmap(xs, xis, z, 
                 title="PINN Heatmap: HBC Interface & Walls",
                 xlabel="Posición Axial x (Normalizada)",
                 ylabel="Coordenada Térmica xi",
                 cmap=:turbo, colorbar_title=" Temp [°C]")
    
    # Líneas guía
    hline!([0], color=:white, linestyle=:dash, label="Interfaz (HBC)")
    hline!([xi_min, xi_max], color=:black, linewidth=2, label="Paredes Adiabáticas")

    # 4. Gráfico de Verificación de HBC (Salto de T en la Interfaz)
    T_inter_h = [f_p(x, 0.0005f0) * DT_phys + Tc_in for x in xs]
    T_inter_c = [f_p(x, -0.0005f0) * DT_phys + Tc_in for x in xs]
    
    p2 = plot(xs, T_inter_h, label="Lado Caliente (xi->0+)", lw=2, color=:red)
    plot!(xs, T_inter_c, label="Lado Frío (xi->0-)", lw=2, color=:blue, linestyle=:dash)
    title!(p2, "Continuidad en la Interfaz (HBC)")
    xlabel!(p2, "x")
    ylabel!(p2, "Temperatura [°C]")

    layout = @layout [a; b]
    final_plot = plot(p1, p2, layout=layout, size=(800, 800))
    display(final_plot)
end


# Ejecutar reporte
generate_final_results(res_final.u)