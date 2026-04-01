
using NeuralPDE, Random, OrdinaryDiffEq, Statistics
import DifferentialEquations as DE
import Sundials
using Lux, OptimizationOptimisers
using Plots

# u[1] = Th, u[2] = Tc, u[3] = U
function intercambiador( du, u, p, t)
    mh, Cph, mc, Cpc, a = p

    # Residuos de las Ecuaciones Diferenciales (Balances de energía)
    res1 =- du[1] - (u[3] * a / (mh * Cph)) * (u[1] - u[2]) 
    res2 =- du[2] + (u[3] * a / (mc * Cpc)) * (u[1] - u[2]) 

    # Residuo de la Ecuación Algebraica: U - f(T) = 0
    res3 = u[3] - (500.0 + 0.1 * (u[1] + u[2])) 
    return [res1, res2, res3]
end

# --- Parámetros y Condiciones Iniciales ---
# Parámetros: [mh, Cph, mc, Cpc, a]
p = [0.5, 4180.0, 0.8, 4180.0, 0.2]

# u₀: Temperaturas iniciales [Th_in, Tc_in, U_inicial]
u₀ = [90.0, 20.0, 511.0] 

# du₀: Derivadas iniciales para que el residuo sea 0 al inicio
# dTh = -(U*a/(mh*Cph))*(Th-Tc)
# dTc = (U*a/(mc*Cpc))*(Th-Tc)
dTh_in = -(511.0 * 0.2 / (0.5 * 4180.0)) * (90.0 - 20.0)
dTc_in = (511.0 * 0.2 / (0.8 * 4180.0)) * (90.0 - 20.0)
du₀ = [dTh_in, dTc_in, 0.0]

# --- Configuración del Problema DAE ---
tspan = (0.0, 10.0) # Tiempo o longitud
differential_vars = [true, true, false] # Th y Tc son dif, U es algebraica

chain = Lux.Chain(
    Lux.Dense(1, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 3)
)

prob = DAEProblem(
    intercambiador,
    du₀,
    u₀,
    tspan,
    p;
    differential_vars = differential_vars
)

opt = OptimizationOptimisers.Adam(0.01)
alg = NNDAE(chain, opt; autodiff = false)

sol = solve(prob, alg, verbose = true, dt = 1 / 100.0, maxiters = 3000, abstol = 1e-8)



######################

function intercambiador2(out, du, u, p, t)
    mh, Cph, mc, Cpc, a = p
    Th, Tc, U = u
    dTh, dTc, dU = du

    # Residuos de las Ecuaciones Diferenciales (Balances de energía)
    # out = (lo que debería ser) - (lo que es)
    out[1] = - (u[3] * a / (mh * Cph)) * (u[1] - u[2]) - du[1]
    out[2] = (u[3] * a / (mc * Cpc)) * (u[1] - u[2]) - du[2]

    # Residuo de la Ecuación Algebraica: U - f(T) = 0
    out[3] = u[3] - (500.0 + 0.1 * (u[1] + u[2])) 
end

prob = DE.DAEProblem(intercambiador2, du₀, u₀, tspan, p, differential_vars=differential_vars)

# --- Solución ---
# Usamos IDA() de Sundials como en tu segundo ejemplo
sol_num = DE.solve(prob, Sundials.IDA(), reltol=1e-8, abstol=1e-8)





# --- 5. Comparación y Visualización ---
# Muestreamos la PINN en los mismos puntos que la solución numérica para comparar
t_eval = range(tspan[1], tspan[2], length=100)
p1 = plot(sol_num, vars=(0, 1), label="Numérico Th", color=:blue, lw=2)
plot!(p1, sol, vars=(0, 1), label="PINN Th", ls=:dash, color=:cyan)

p2 = plot(sol_num, vars=(0, 2), label="Numérico Tc", color=:red, lw=2)
plot!(p2, sol, vars=(0, 2), label="PINN Tc", ls=:dash, color=:orange)

p3 = plot(sol_num, vars=(0, 3), label="Numérico U", color=:green, lw=2)
plot!(p3, sol, vars=(0, 3), label="PINN U", ls=:dash, color=:lime)

plot(p1, p2, p3, layout=(3,1), size=(800, 800), xlabel="Tiempo (o Longitud)")
