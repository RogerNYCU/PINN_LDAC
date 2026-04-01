using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, 
      OptimizationOptimisers, Plots, Printf, Zygote, Statistics, DomainSets
import ModelingToolkit: Interval, Differential
using Random

# --- 1. PARÁMETROS FÍSICOS ---
Lx, h_phys = 2.0, 0.05
kh, kc = 1.0, 1.1
rho, cp, um = 1000.0, 4180.0, 0.05
Th_in, Tc_in = 80.0, 20.0

# Constantes Normalizadas para la PDE
C_h = ((kh / (rho * cp)) * Lx) / (um * h_phys^2)
C_c = ((kc / (rho * cp)) * Lx) / (um * h_phys^2)

@parameters x eta
@variables T(..)
Dx = Differential(x)
Detaeta = Differential(eta)^2

# Funciones auxiliares matemáticas (sin condicionales lógicos)
y_norm(e) = (e * (kh + kc) + abs(e) * (kh - kc)) / 2
u_vel(e) = 1.5 * (1.0 - y_norm(e)^2)
C_eff(e) = ( (1.0 + sign(e))/2 * C_h + (1.0 - sign(e))/2 * C_c )

# --- 2. DEFINICIÓN DEL SISTEMA ---
eq = u_vel(eta) * Dx(T(x, eta)) ~ C_eff(eta) * Detaeta(T(x, eta))
domains = [x ∈ Interval(0.0, 1.0), eta ∈ Interval(-1.0/kc, 1.0/kh)]
bcs = [Differential(eta)(T(x, -1.0/kc)) ~ 0.0, 
       Differential(eta)(T(x, 1.0/kh)) ~ 0.0]

# Ansatz HBC: Garantiza condiciones de entrada exactas
function hbc_ansatz(phi, θ, p)
    return (x_v, eta_v) -> begin
        nn = phi([x_v, eta_v], θ)[1]
        s = 1.0 / (1.0 + exp(-10.0 * eta_v)) 
        base = s * 1.0 + (1.0 - s) * 0.0
        dist = s * x_v + (1.0 - s) * (x_v - 1.0)
        return base + dist * nn
    end
end

# --- 3. DISCRETIZACIÓN Y ENSAMBLAJE DE PÉRDIDA ---
# Red más capaz para capturar la transición en la interfaz
chain = Lux.Chain(
    Lux.Dense(2, 32, Lux.sin), 
    Lux.Dense(32, 32, Lux.sin), 
    Lux.Dense(32, 32, Lux.sin), 
    Lux.Dense(32, 1)
)
Random.seed!(42)
strategy = NeuralPDE.QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy; custom_strategy = hbc_ansatz)

@named pde_system = PDESystem(eq, bcs, domains, [x, eta], [T(x, eta)])
sym_prob = symbolic_discretize(pde_system, discretization)

# Construcción manual de la función de optimización
# Sumamos las pérdidas de la PDE y de las Condiciones de Contorno (BCs)
pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

callback = function (p, l)
    # Usamos una variable global o de scope superior para contar iteraciones
    if !@isdefined(iter_count)
        global iter_count = 0
    end
    global iter_count += 1
    
    if iter_count % 100 == 0
        println("Iteración: $iter_count | Pérdida Total: $l")
    end
    
    return false # Retornar true detendría el entrenamiento prematuramente
end

# loss_func = OptimizationFunction((θ, p) -> begin
#     l_pde = sum(f -> mean(abs2, f(θ)), pde_loss_functions)
#     l_bc = sum(f -> mean(abs2, f(θ)), bc_loss_functions)
#     return l_pde + l_bc
# end, Optimization.AutoZygote())
# En tu bucle de pérdida manual, añade un multiplicador para la zona de la interfaz
loss_func = OptimizationFunction((θ, p) -> begin
    # Evaluamos la PDE
    l_pde = mean(abs2, pde_loss_functions[1](θ))
    
    # Bonus: Penalizar si la derivada en la interfaz es cero 
    # (Para forzar a la red a que "pase" calor)
    return l_pde 
end, Optimization.AutoZygote())
prob = OptimizationProblem(loss_func, sym_prob.flat_init_params)

# --- 4. ENTRENAMIENTO ---
println("Iniciando Adam...")
res = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 1000, callback = callback)

println("Iniciando LBFGS para refinamiento...")
prob2 = OptimizationProblem(loss_func, res.u)
res = solve(prob2, LBFGS(); maxiters = 500, callback = callback)

# --- 5. RESULTADOS Y BALANCE ---
T_func = hbc_ansatz(discretization.phi, res.u, nothing)

function post_procesado_y_balance(T_f)
    # 1. Heatmap Desnormalizado


    # 2. Balance de Energía
    n_int = 150
    # Canal Caliente (eta > 0)
    e_h = range(0, 1.0/kh, length=n_int); de_h = (1.0/kh)/n_int
    Q_h_in = sum([u_vel(e) * 1.0 * de_h for e in e_h])
    Q_h_out = sum([u_vel(e) * T_f(1.0, e) * de_h for e in e_h])

    # Canal Frío (eta < 0)
    e_c = range(-1.0/kc, 0, length=n_int); de_c = (1.0/kc)/n_int
    Q_c_in = sum([u_vel(e) * 0.0 * de_c for e in e_c])
    Q_c_out = sum([u_vel(e) * T_f(0.0, e) * de_c for e in e_c])

    factor = rho * cp * um * h_phys
    dh = abs(Q_h_in - Q_h_out) * factor
    dc = abs(Q_c_out - Q_c_in) * factor
    error_balance = round(abs(dh - dc)/dh * 100, digits=4)
    println("\n" * "="^30)
    println("BALANCE DE ENERGÍA (Watts/m)")
    println("-"^30)
    println("Q Cedido (Hot):  $(round(dh, digits=2)) W/m")
    println("Q Ganado (Cold): $(round(dc, digits=2)) W/m")
    println("Error Balance:   $(error_balance) %")
    println("="^30)
    xs = 0.0:0.02:1.0
    etas = range(-1.0/kc, 1.0/kh, length=100)
    T_map = [(T_f(xv, ev)*(Th_in-Tc_in)+Tc_in) for ev in etas, xv in xs]
    y_phys_axis = [(ev > 0 ? ev*kh : ev*kc)*h_phys for ev in etas]
    
    p1 = heatmap(xs .* Lx, y_phys_axis, T_map, c=:turbo, 
                 title="PINN HBC: Physical Temperature $error_balance %, $(round(dh, digits=2)) ", xlabel="x (m)", ylabel="y (m)")
    display(p1)
    savefig(p1, "remember/25Hx12.png")
end

post_procesado_y_balance(T_func)
