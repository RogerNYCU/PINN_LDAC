#aqui retomamos la estrategia de coordenada térmica 
#hard para las entradas, pero soft para las paredes adiabaticas.
#es lento por fowardiff, la version multihilo esta al final
#en este punto inicie con multihilo, es la conversacion  Pnn para intercambiador de calor

using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, DomainSets
using OptimizationOptimisers, Plots, Printf, Statistics, ComponentArrays, Random, ForwardDiff

# ==========================================
# 1. PARÁMETROS FÍSICOS Y TRANSFORMACIÓN
# ==========================================
const Lx, h_phys = 2.0, 0.05
const kh, kc = 1.0, 1.1
const rho, cp, um = 1000.0, 4180.0, 0.05
const Th_in, Tc_in = 80.0, 20.0

const xi_min = -h_phys / kc
const xi_max =  h_phys / kh
const DT_phys = Th_in - Tc_in

# ==========================================
# 2. ARQUITECTURA Y ANSATZ HBC
# ==========================================
Random.seed!(42)
chain = Lux.Chain(
    Lux.Dense(2, 32, Lux.sin), 
    Lux.Dense(32, 32, Lux.sin), 
    Lux.Dense(32, 1)
)

ps, st = Lux.setup(Random.default_rng(), chain)
θ_flat = ComponentVector(ps)

step_smooth(xi) = 0.5 * (1 + tanh(100.0 * xi))

function hbc_ansatz(x, xi, p)
    nn = chain([x, xi], p, st)[1][1]
    s = step_smooth(xi)
    # HBC: Normalizado 1.0 (Hot) y 0.0 (Cold)
    t_in = s * 1.0 + (1.0 - s) * 0.0
    dist = s * x + (1.0 - s) * (x - 1.0)
    return t_in + dist * nn
end

# ==========================================
# 3. PDE Y PÉRDIDA
# ==========================================
# Reducimos discretización para mayor velocidad de iteración
const xs_train = range(0.0, 1.0, length=30)
const xis_train = range(xi_min, xi_max, length=20)

function u_vel_xi(xi_v)
    eta = xi_v > 0 ? (xi_v * kh / h_phys) : (abs(xi_v) * kc / h_phys)
    return um * 1.5 * (1.0 - eta^2) # Corregido: um incluido
end

function loss_function(p, _)
    l_pde = sum(Iterators.product(xs_train, xis_train)) do (x, xi)
        # Derivadas respecto a la física
        Tx = ForwardDiff.derivative(vx -> hbc_ansatz(vx, xi, p), x)
        Txi_xi = ForwardDiff.derivative(vxi -> ForwardDiff.derivative(vvxi -> hbc_ansatz(x, vvxi, p), vxi), xi)
        
        u_eff = xi > 0 ? u_vel_xi(xi) : -u_vel_xi(xi)
        
        # Balance: rho*cp * u * dT/dx = (1/Lx) * d^2T/dxi^2
        # (Nota: El 1/Lx aparece al normalizar la coordenada x de 0-Lx a 0-1)
        abs2( (rho * cp) * u_eff * Tx - (1.0/Lx) * Txi_xi )
    end / (30 * 20)

    l_wall = sum(xs_train) do x
        abs2(ForwardDiff.derivative(vxi -> hbc_ansatz(x, vxi, p), xi_min)) +
        abs2(ForwardDiff.derivative(vxi -> hbc_ansatz(x, vxi, p), xi_max))
    end / 30

    return l_pde + 10.0 * l_wall
end

# ==========================================
# 4. ENTRENAMIENTO (CAMBIO DE AD A AutoForwardDiff)
# ==========================================
println(">>> Iniciando Entrenamiento...")
callback = function (p,l)
    if !@isdefined(iter_count) global iter_count =0 end
    global iter_count +=1
    if iter_count % 100 == 0 println("iteration: $iter_count | loss: $l") end
    return false
end

# Cambiamos AutoZygote por AutoForwardDiff para evitar el error de iterate(nothing)
optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
prob = OptimizationProblem(optf, θ_flat)

println("Fase 1: Adam")
res = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 1000)

println("Fase 2: LBFGS")
prob2 = OptimizationProblem(optf, res.u)
res = solve(prob2, LBFGS(); maxiters = 400)

# ==========================================
# 5. REPORTE FINAL
# ==========================================
function final_report(p)
    xis_h = range(1e-5, xi_max, length=100)
    xis_c = range(xi_min, -1e-5, length=100)
    dxi_h = xi_max / 100
    dxi_c = abs(xi_min) / 100
    
    # Calor en Watts/m: ∫ rho * cp * u * T * (DT_phys) dy
    # Como integramos en dxi, y dy = k * dxi, el factor es rho * cp * DT_phys
    factor = rho * cp * DT_phys
    
    q_h_in  = sum(xi -> u_vel_xi(xi) * hbc_ansatz(0.0, xi, p) * dxi_h, xis_h) 
    q_h_out = sum(xi -> u_vel_xi(xi) * hbc_ansatz(1.0, xi, p) * dxi_h, xis_h)
    Qh = (q_h_in - q_h_out) * factor
    
    q_c_in  = sum(xi -> u_vel_xi(xi) * hbc_ansatz(1.0, xi, p) * dxi_c, xis_c)
    q_c_out = sum(xi -> u_vel_xi(xi) * hbc_ansatz(0.0, xi, p) * dxi_c, xis_c)
    Qc = (q_c_out - q_c_in) * factor
    error_balance=round(abs(Qh - Qc)/max(abs(Qh), 1.0) * 100, digits=4)
    println("\n========================================")
    println("   HEAT BALANCE REPORT (AutoForwardDiff)")
    println("----------------------------------------")
    println("Heat Released (Hot):  $(round(Qh, digits=2)) W/m")
    println("Heat Gained (Cold):   $(round(Qc, digits=2)) W/m")
    println("Balance Error:        $error_balance %")
    println("========================================\n")

    # Visualización
    xs_p = range(0.0, 1.0, length=50)
    xis_p = range(xi_min, xi_max, length=50)
    z = [hbc_ansatz(x, xi, p) * DT_phys + Tc_in for xi in xis_p, x in xs_p]
    heatmap(xs_p, xis_p, z, title="PINN HBC: $error_balance %, $(round(Qc, digits=2)) ", cmap=:turbo)
end

final_report(res.u)