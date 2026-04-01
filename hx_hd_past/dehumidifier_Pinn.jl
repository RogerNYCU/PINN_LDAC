using NeuralPDE, Lux, Optimization, OptimizationOptimisers
using OrdinaryDiffEq, Plots, Random

# --- 1. DAE System Definition (LDAC Physics) ---
# u[1]=Ta, u[2]=ωa, u[3]=Ts, u[4]=ξw, u[5]=ω_int, u[6]=dq, u[7]=dm
function ldac_pinn_system(du, u, p, x)
    # Variable unpacking
    Ta, ωa, Ts, ξw, ω_int, dq, dm = u
    dTa, dωa, dTs, dξw, _, _, _ = du
    
    # Dynamic properties (using functions from props_regression.jl)
    Pv_sol = _Pᵥₐₚₒᵣ_ₛₒₗ(Ts, ξw)
    cp_s   = _cpₛₒₗ(Ts, ξw)
    cp_a   = _cpₐ(Ta, ωa)
    h_evap = i_fg(Ts)
    
    # Coefficients (simplified for network stability)
    ht, hm = 35.0, 0.035 

    # Differential Residuals
    res1 = dTa - (-dq / (ṁ_a * cp_a))
    res2 = dωa - (-dm / ṁ_a)
    res3 = dTs - ((-dq + dm * h_evap) / (ṁ_s * cp_s))
    res4 = dξw - (-dm / ṁ_s)
    
    # Algebraic Residuals (must equal 0)
    res5 = ω_int - (0.62185 * Pv_sol / (101325.0 - Pv_sol))
    res6 = dq - (ht * FD * (Ta - Ts))
    res7 = dm - (hm * FD * (ωa - ω_int))
    
    return [res1, res2, res3, res4, res5, res6, res7]
end

# --- 2. Neural Network Configuration ---
rng = Random.default_rng()
# Input: x (1), Output: u (7 variables)
chain = Lux.Chain(
    Lux.Dense(1, 64, Lux.tanh),
    Lux.Dense(64, 64, Lux.tanh),
    Lux.Dense(64, 64, Lux.tanh),
    Lux.Dense(64, 7)
)

# --- 3. Parameters and Boundary Conditions ---
tspan = (0.0, H_total)
# Note: For NNDAE boundary value problems, u0 are the inputs at x=0
# Air enters at x=0, Solution "exits" at x=0 (estimated)
u₀ = [Ta_in, ωa_in, 24.8 + 273.15, 0.725, 0.015, 20.0, 0.0001]
du₀ = zeros(7)
differential_vars = [true, true, true, true, false, false, false]

prob = DAEProblem(
    ldac_pinn_system,
    du₀,
    u₀,
    tspan,
    differential_vars = differential_vars
)

# --- 4. PINN Training ---
# NNDAE will use the network structure to approximate the DAE solution
opt = OptimizationOptimisers.Adam(0.005)
alg = NNDAE(chain, opt; autodiff = false)

println("Training PINN for LDAC system...")
sol = solve(prob, alg, verbose = true, dt = H_total/100, maxiters = 5000)

# --- 5. Post-processing and Visualization ---
# Generate inspection points
x_test = collect(0.0:H_total/50:H_total)
u_pred = [sol(x) for x in x_test]

# Extract profiles
Ta_p = [u[1]-273.15 for u in u_pred]
Ts_p = [u[3]-273.15 for u in u_pred]
wa_p = [u[2] for u in u_pred]
xw_p = [u[4] for u in u_pred]

p1 = plot(x_test, [Ta_p Ts_p], label=["Air T" "Sol T"], title="PINN Results", ylabel="T [°C]")
p2 = plot(x_test, wa_p, label="Air ω", color=:green, ylabel="ω [kg/kg]")
p3 = plot(x_test, xw_p, label="Water ξ", color=:blue, xlabel="Height [m]", ylabel="Mass Fraction")

display(plot(p1, p2, p3, layout=(3,1), size=(800, 900)))

# --- 6. PINN Mass Balance ---
water_air = ṁ_a * (wa_p[1] - wa_p[end])
water_sol  = ṁ_s * (xw_p[1] - xw_p[end])
println("PINN Balance - Error: ", abs(water_air - water_sol)/water_air * 100, " %")