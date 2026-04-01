#no funciona en general

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, Plots, Printf, DomainSets
using OptimizationOptimisers, Statistics, LineSearches
using BenchmarkTools




# --- 1. PARÁMETROS FÍSICOS ---
Lx, h_phys = 2.0, 0.05
rho, cp = 1000.0, 4180.0
kh, kc = 1.0, 1.1
um = 0.001
Th_in, Tc_in = 80.0, 20.0

alpha_h = kh / (rho * cp)
alpha_c = kc / (rho * cp)
Pe_h = (um * h_phys^2) / (alpha_h * Lx)
Pe_c = (um * h_phys^2) / (alpha_c * Lx)

@parameters x eta
@variables Th(..) Tc(..)
Dx = Differential(x)
Deta = Differential(eta)
Deta2 = Differential(eta)^2

u_star(e) = 2.0 * abs(e) * (1.0 - abs(e))
plot(range(-1, 1, 100), u_star.(range(-1, 1, 100)), label="u_star(η)", xlabel="η", ylabel="u_star", title="Velocity Profile", lw=2)

# --- 2. SISTEMA ---
eqs = [
    u_star(eta) * Dx(Th(x, eta)) ~ (1.0/Pe_h) * Deta2(Th(x, eta)),
   -u_star(eta) * Dx(Tc(x, eta)) ~ (1.0/Pe_c) * Deta2(Tc(x, eta))
]

domains = [x ∈ Interval(0.0, 1.0), eta ∈ Interval(-1.0, 1.0)]

bcs = [
    Deta(Th(x, 1.0)) ~ 0.0,   
    Deta(Tc(x, -1.0)) ~ 0.0,  
    Th(x, 0.0) ~ Tc(x, 0.0),  
    kh * Deta(Th(x, 0.0)) ~ kc * Deta(Tc(x, 0.0)) 
]

@named pde_system = PDESystem(eqs, bcs, domains, [x, eta], [Th(x, eta), Tc(x, eta)])

# --- 3. RED Y ANSATZ HBC ---
input_dim = 2
hidden_dim = 32
chain_h = Lux.Chain(Lux.Dense(input_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, 1))
chain_c = Lux.Chain(Lux.Dense(input_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, hidden_dim, Lux.tanh), Lux.Dense(hidden_dim, 1))
function hbc_custom_strategy(phi, θ, p)

    return [

        (x_v, eta_v) -> 1.0 + x_v * phi[1]([x_v, eta_v], θ.depvar.Th)[1],

        (x_v, eta_v) -> 0.0 + (x_v - 1.0) * phi[2]([x_v, eta_v], θ.depvar.Tc)[1]

    ]

end

# --- 3. MEJORA: ESTRATEGIA DE MUESTREO REFORZADO ---
# Añadimos puntos específicos en la interfaz (eta=0) para obligar a las redes a tocarse
# Usamos GridTraining para los bordes y QuasiRandom para el dominio
discretization = PhysicsInformedNN([chain_h, chain_c], 
                                    QuasiRandomTraining(1500); 
                                    custom_strategy = hbc_custom_strategy)

sym_prob = symbolic_discretize(pde_system, discretization)

# --- 4. FUNCIÓN DE PÉRDIDA CON PESOS DINÁMICOS ---
pde_loss_f = sym_prob.loss_functions.pde_loss_functions
bc_loss_f = sym_prob.loss_functions.bc_loss_functions

loss_func = OptimizationFunction((θ, p) -> begin
    l_pde = sum(f -> mean(abs2, f(θ)), pde_loss_f)
    
    # BCs Externas (Adiabáticas)
    l_bc_ext = mean(abs2, bc_loss_f[1](θ)) + mean(abs2, bc_loss_f[2](θ))
    
    # Interfaz (Continuidad T y Flujo) - SUBIMOS EL PESO A 2000.0
    l_interface = mean(abs2, bc_loss_f[3](θ)) + mean(abs2, bc_loss_f[4](θ))
    
    # Penalizamos más fuerte la interfaz para colapsar el abismo térmico
    return l_pde + l_bc_ext + 2000.0 * l_interface 
end, Optimization.AutoZygote())

opt_prob = OptimizationProblem(loss_func, sym_prob.flat_init_params)

# --- 5. ENTRENAMIENTO REFORZADO ---
println("Fase 1: Adam (Fijando la interfaz)...")
@time res = solve(opt_prob, OptimizationOptimisers.Adam(0.005); maxiters = 1500)

# Actualizamos el problema con los nuevos pesos y la posición de Adam
println("Fase 2: LBFGS (Refinamiento de balance)...")
opt_prob2 = OptimizationProblem(loss_func, res.u)
@time res = solve(opt_prob2, LBFGS(linesearch = BackTracking()); 
            maxiters = 1200, 
            allow_f_increases = true, 
            )

# --- 6. DIAGNÓSTICO POST-MEJORA ---
# Ejecuta de nuevo tu función 'post_balance' y 'diagnostico_interfaz'
function post_balance(res, discretization)
    phi = discretization.phi
    u_params = res.u
    DeltaT = Th_in - Tc_in
    
    # Temperaturas desnormalizadas
    Th_f(x, e) = (1.0 + x * phi[1]([x, e], u_params.depvar.Th)[1]) * DeltaT + Tc_in
    Tc_f(x, e) = (0.0 + (x - 1.0) * phi[2]([x, e], u_params.depvar.Tc)[1]) * DeltaT + Tc_in

    detas = range(0.0, 1.0, length=200); de = 1.0/199
    factor = rho * cp * um * h_phys

    # El calor cedido (Hot) es la energía al entrar (x=0) menos al salir (x=1)
    energy_h_in = sum(u_star(e) * Th_f(0.0, e) for e in detas) * de
    energy_h_out = sum(u_star(e) * Th_f(1.0, e) for e in detas) * de
    Qh = (energy_h_in - energy_h_out) * factor
    
    # El calor ganado (Cold) es la energía al salir (x=0) menos al entrar (x=1)
    energy_c_out = sum(u_star(e) * Tc_f(0.0, -e) for e in detas) * de
    energy_c_in = sum(u_star(e) * Tc_f(1.0, -e) for e in detas) * de
    Qc = (energy_c_out - energy_c_in) * factor

    println("\n" * "="^40)
    @printf("Q Cedido: %.2f W/m\n", Qh)
    @printf("Q Ganado: %.2f W/m\n", Qc)
    @printf("Error Balance: %.4f %%\n", abs(Qh - Qc)/max(abs(Qh), abs(Qc)) * 100)
    println("="^40)
end
post_balance(res, discretization)

phi = discretization.phi
T_h_f(x_v, e_v) = 1.0 + x_v * phi[1]([x_v, e_v], res.u.depvar.Th)[1]
T_c_f(x_v, e_v) = 0.0 + (x_v - 1.0) * phi[2]([x_v, e_v], res.u.depvar.Tc)[1]

# Visualización comparativa
xs = 0.0:0.05:1.0
etas_h = 0.0:0.05:1.0
etas_c = -1.0:0.05:0.0

th_map = [T_h_f(xv, ev) for ev in etas_h, xv in xs]
tc_map = [T_c_f(xv, ev) for ev in etas_c, xv in xs]

p1 = heatmap(xs, etas_h, th_map, title="Canal Caliente (HBC)", cmap=:thermal)
p2 = heatmap(xs, etas_c, tc_map, title="Canal Frío (HBC)", cmap=:thermal)
plot( p2, p1, layout=(2,1))


function diagnostico_interfaz(res, discretization)
    phi = discretization.phi
    u_params = res.u
    
    # Rango de x para evaluar la interfaz
    xs = range(0.0, 1.0, length=100)
    eta_int = 0.0
    
    # 1. Temperaturas en la interfaz (Deberían ser idénticas)
    Th_int = [1.0 + x * phi[1]([x, eta_int], u_params.depvar.Th)[1] for x in xs]
    Tc_int = [0.0 + (x - 1.0) * phi[2]([x, eta_int], u_params.depvar.Tc)[1] for x in xs]
    
    p1 = plot(xs, [Th_int Tc_int], 
              label=["Canal Caliente (eta=0)" "Canal Frío (eta=0)"],
              title="Continuidad de Temperatura en Interfaz",
              xlabel="x (m)", ylabel="T normalizada", lw=2)

    # 2. Gradientes Térmicos (Flujo de Calor)
    # Calculamos la derivada numérica respecto a eta en 0
    deps = 1e-4
    grad_h = [( (1.0 + x * phi[1]([x, eta_int+deps], u_params.depvar.Th)[1]) - 
                (1.0 + x * phi[1]([x, eta_int], u_params.depvar.Th)[1]) ) / deps for x in xs]
    
    grad_c = [( (0.0 + (x-1.0) * phi[2]([x, eta_int], u_params.depvar.Tc)[1]) - 
                (0.0 + (x-1.0) * phi[2]([x, eta_int-deps], u_params.depvar.Tc)[1]) ) / deps for x in xs]
    
    # Aplicamos conductividades: kh * grad_h ≈ kc * grad_c
    flux_h = kh .* grad_h
    flux_c = kc .* grad_c

    p2 = plot(xs, [flux_h flux_c], 
              label=["Flujo Hot (kh * dT/deta)" "Flujo Cold (kc * dT/deta)"],
              title="Continuidad de Flujo (Ley de Fourier)",
              xlabel="x (m)", ylabel="Flujo de Calor Local", lw=2, linestyle=[:solid :dash])

    display(plot(p1, p2, layout=(2,1), size=(800, 700)))
end

diagnostico_interfaz(res, discretization)


using ForwardDiff

function mapa_residuos(res, discretization)
    phi = discretization.phi
    u_p = res.u
    
    # Definir funciones Th y Tc para ForwardDiff
    f_h(X) = 1.0 + X[1] * phi[1](X, u_p.depvar.Th)[1]
    f_c(X) = 0.0 + (X[1] - 1.0) * phi[2](X, u_p.depvar.Tc)[1]

    xs = range(0.1, 0.9, length=40) # Evitamos bordes exactos por estabilidad numérica
    etas_h = range(0.1, 0.9, length=40)
    
    residuos = zeros(length(etas_h), length(xs))

    for (i, e) in enumerate(etas_h), (j, x) in enumerate(xs)
        X = [x, e]
        # Gradientes automáticos
        grad = ForwardDiff.gradient(f_h, X) # [dT/dx, dT/deta]
        hess = ForwardDiff.hessian(f_h, X)   # [d2T/dx2, d2T/dx d_eta; ...]
        
        # Residuo = u*dT/dx - (1/Pe)*d2T/deta2
        residuos[i,j] = abs(u_star(e) * grad[1] - (1/Pe_h) * hess[2,2])
    end

    heatmap(xs, etas_h, residuos, title="Residuos PDE: ¿Dónde falla la física?",
            xlabel="x", ylabel="eta", cmap=:viridis)
end

mapa_residuos(res, discretization)