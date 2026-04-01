using LinearAlgebra
using Plots
using Interpolations
using StaticArrays
using NonlinearSolve
using CoolProp
using Statistics
include("props_regression.jl")
# --- LOADING PROPERTIES (Simulating include("props_regression.jl")) ---
# All your functions go here: _Pᵥₐₚₒᵣ_ₛₒₗ, _ρₛₒₗ, _cpₛₒₗ, _iₛₒₗ, _ρₐ, etc.
# [It is assumed that the provided functions are loaded in the environment]

function solve_LDAC_Professional()
    # --- 1. CONFIGURATION AND MESH ---
    L, delta_a, delta_s = 1.0, 0.005, 0.001
    Nx, Ny_a, Ny_s = 60, 20, 10
    dx, dy_a, dy_s = L/Nx, delta_a/Ny_a, delta_s/Ny_s

    # --- 2. INLET CONDITIONS (Convert to Kelvin for functions) ---
    Ta_in_C, wa_in = 35.0, 0.025
    Ts_in_C, X_in = 18.0, 0.35  # X_in as LiCl fraction (0.35)
    
    T_ref = 273.15
    Ta_in, Ts_in = Ta_in_C + T_ref, Ts_in_C + T_ref
    u_a, u_s = 1.5, 0.15

    # --- 3. INITIALIZATION ---
    Ta = fill(Ta_in, Nx, Ny_a)
    wa = fill(wa_in, Nx, Ny_a)
    Ts = fill(Ts_in, Nx, Ny_s)
    X  = fill(X_in, Nx)

    # --- 4. SOLVER ---
    for iter in 1:800
        Ta_old = copy(Ta)
        
        # A. AIR (Flow from i=1 to Nx)
        for i in 1:Nx
            # Air properties in the section
            curr_Ta_avg = mean(Ta[i,:])
            curr_wa_avg = mean(wa[i,:])
            rho_a = _ρₐ(curr_Ta_avg, curr_wa_avg)
            cp_a  = _cpₐ(curr_Ta_avg, curr_wa_avg)
            k_a   = _kₐ(curr_Ta_avg, curr_wa_avg)
            D_va  = 2.6e-5 # Approximate diffusivity if not in props

            # Equilibrium humidity at interface (LiCl)
            # Pv = _Pᵥₐₚₒᵣ_ₛₒₗ(T_sol, X) -> wa_eq
            T_int = Ts[i, Ny_s]
            Pv_sat = _Pᵥₐₚₒᵣ_ₛₒₗ(T_int, X[i])
            wa_eq = 0.622 * Pv_sat / (101325.0 - Pv_sat)

            for j in 1:Ny_a
                T_west = (i == 1) ? Ta_in : Ta[i-1, j]
                w_west = (i == 1) ? wa_in : wa[i-1, j]

                F_heat_a = (k_a * dx) / (rho_a * u_a * cp_a * dy_a^2)
                F_mass_a = (D_va * dx) / (u_a * dy_a^2)

                T_south = (j == 1) ? Ts[i, Ny_s] : Ta[i, j-1]
                T_north = (j == Ny_a) ? Ta[i, j] : Ta[i, j+1]
                
                w_south = (j == 1) ? wa_eq : wa[i, j-1]
                w_north = (j == Ny_a) ? wa[i, j] : wa[i, j+1]

                Ta[i,j] = (T_west + F_heat_a*(T_north + T_south)) / (1 + 2*F_heat_a)
                wa[i,j] = (w_west + F_mass_a*(w_north + w_south)) / (1 + 2*F_mass_a)
            end
        end

        # B. SOLUTION (Counter-current: i from Nx to 1)
        for i in Nx:-1:1
            # Solution properties
            curr_Ts_avg = mean(Ts[i,:])
            rho_s = _ρₛₒₗ(curr_Ts_avg, X[i])
            cp_s  = _cpₛₒₗ(curr_Ts_avg, X[i])
            k_s   = _𝑘ₛₒₗ(curr_Ts_avg, X[i])
            
            # Mass flux at interface
            dw_dy_int = (wa[i,2] - wa[i,1]) / dy_a 
            m_flux = _ρₐ(Ta[i,1], wa[i,1]) * 2.6e-5 * dw_dy_int
            
            # Latent heat (i_fg from your correlations)
            h_fg_local = i_fg(Ts[i, Ny_s])

            for j in 1:Ny_s
                T_east = (i == Nx) ? Ts_in : Ts[i+1, j]
                F_sol = (k_s * dx) / (rho_s * u_s * cp_s * dy_s^2)

                T_bot = (j == 1) ? Ts[i, j] : Ts[i, j-1] # Adiabatic wall or fixed
                T_top = (j == Ny_s) ? Ta[i, 1] : Ts[i, j+1]
                
                S_latent = (j == Ny_s) ? (m_flux * h_fg_local * dx) / (rho_s * u_s * cp_s * dy_s) : 0.0
                
                Ts[i,j] = (T_east + F_sol*(T_top + T_bot) + S_latent) / (1 + 2*F_sol)
            end

            # C. MASS BALANCE (Update X)
            m_dot_s = rho_s * u_s * delta_s
            X_prev = (i == Nx) ? X_in : X[i+1]
            X[i] = (m_dot_s * X_prev) / (m_dot_s + m_flux * dx)
        end

        if maximum(abs.(Ta - Ta_old)) < 1e-4 break end
    end

    # --- 5. GLOBAL ENERGY BALANCE CALCULATION ---
    # Air: m_a * (h_in - h_out)
    h_a_in = CoolProp.HAPropsSI("H", "T", Ta_in, "P", 101325.0, "W", wa_in)
    Ta_out_avg = mean(Ta[Nx, :])
    wa_out_avg = mean(wa[Nx, :])
    h_a_out = CoolProp.HAPropsSI("H", "T", Ta_out_avg, "P", 101325.0, "W", wa_out_avg)
    
    rho_a_in = _ρₐ(Ta_in, wa_in)
    m_dot_a = rho_a_in * u_a * delta_a
    Q_air = m_dot_a * (h_a_in - h_a_out)

    # Solution: m_s_out * h_s_out - m_s_in * h_s_in
    # Note: We use the enthalpies _iₛₒₗ from your correlations
    h_s_in = _iₛₒₗ(Ts_in, X_in)
    Ts_out_avg = mean(Ts[1, :])
    h_s_out = _iₛₒₗ(Ts_out_avg, X[1])
    
    m_dot_s_in = _ρₛₒₗ(Ts_in, X_in) * u_s * delta_s
    m_dot_s_out = m_dot_s_in + (m_dot_a * (wa_in - wa_out_avg)) # Global mass balance
    Q_sol = (m_dot_s_out * h_s_out) - (m_dot_s_in * h_s_in)

    return Ta .- T_ref, wa, Ts .- T_ref, X, Q_air, Q_sol
end

# Execution
Ta, wa, Ts, X, Qa, Qs = solve_LDAC_Professional()

println("--- Energy Balance ---")
println("Heat released by air: ", round(Qa, digits=2), " W/m")
println("Heat absorbed by solution: ", round(Qs, digits=2), " W/m")
println("Balance error: ", round(abs(Qa - Qs)/Qa * 100, digits=2), "%")

# --- Visualization ---
l = @layout [a ; b ; c ; d]
p1 = heatmap(1:60, (1:20).*0.25, Ta', c=:thermal, title="Air Temperature Profile (°C)", ylabel="y (mm)")
p3 = heatmap(1:60, (1:20).*0.25, wa', c=:viridis, title="Air Humidity Profile (kg/kg)", ylabel="y (mm)")
p2 = heatmap(1:60, (1:10).*0.1, Ts', c=:thermal, title="Solution Temperature Profile (°C)", ylabel="y (mm)")
p4 = plot(X, lw=3, title="Solution Concentration (X)", xlabel="X Position", color=:red, label="LiCl")
plot(p1, p2, p3, p4, layout=l, size=(800,1000))
