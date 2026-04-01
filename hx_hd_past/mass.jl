using LinearAlgebra
using Plots
const CP_VAPOR = 1860.0


    # --- 1. MESH AND GEOMETRY ---
    L, delta_a, delta_s = 1.0, 0.005, 0.001
    Nx, Ny_a, Ny_s = 60, 20, 10
    dx, dy_a, dy_s = L/Nx, delta_a/Ny_a, delta_s/Ny_s

    # --- 2. PROPERTIES ---
    u_a, u_s = 1.5, 0.15 
    rho_a, cp_a, k_a, D_va = 1.2, 1006.0, 0.026, 2.5e-5
    rho_s, cp_s, k_s = 1250.0, 3100.0, 0.6
    H_FG = 2450e3 # Latent heat + dilution heat
    
    # --- 3. INITIALIZATION ---
    Ta = fill(35.0, Nx, Ny_a); wa = fill(0.025, Nx, Ny_a)
    Ts = fill(18.0, Nx, Ny_s); X = fill(0.35, Nx)
    
    Ta_in, wa_in = 35.0, 0.025
    Ts_in, X_in = 18.0, 0.35

function solve_LDAC_2D_Improved()


    # --- 4. SOLVER ---
    for iter in 1:1200
        Ta_old, wa_old = copy(Ta), copy(wa)
        
        # A. AIR (Direction i: 1 -> Nx)
        for i in 1:Nx
            # Interface: j=1 of air contacts j=Ny_s of solution
            # Calculate local equilibrium humidity (based on T and X at interface)
            # Simplified LiCl model: P_sat_pure * f(X)
            T_int = Ts[i, Ny_s]
            w_eq = (0.0007 * T_int^2 - 0.015 * T_int + 0.2) * (1.0 - X[i]) * 0.01 
            # Readjust w_eq to be physically coherent (simplified)
            w_eq = max(0.005, 0.0005 * T_int * (1.1 - X[i])) 

            for j in 1:Ny_a
                T_west = (i == 1) ? Ta_in : Ta[i-1, j]
                w_west = (i == 1) ? wa_in : wa[i-1, j]
                
                # Diffusion in Y (Air)
                F_heat_a = (k_a * dx) / (rho_a * u_a * cp_a * dy_a^2)
                F_mass_a = (D_va * dx) / (u_a * dy_a^2)
                
                # Boundary conditions in Y for Air
                T_south = (j == 1) ? Ts[i, Ny_s] : Ta[i, j-1] # Contact with solution
                T_north = (j == Ny_a) ? Ta[i, j] : Ta[i, j+1] # Adiabatic above
                
                w_south = (j == 1) ? w_eq : wa[i, j-1]
                w_north = (j == Ny_a) ? wa[i, j] : wa[i, j+1]
                
                Ta[i,j] = (T_west + F_heat_a*(T_north + T_south)) / (1 + 2*F_heat_a)
                wa[i,j] = (w_west + F_mass_a*(w_north + w_south)) / (1 + 2*F_mass_a)
            end
        end

        # B. SOLUTION (Counterflow: i from Nx to 1)
        for i in Nx:-1:1
            # Calculate Mass Flux (m'') via real gradient at interface
            # Forward derivative of 2nd order for better precision at boundary
            dw_dy_int = (-3*wa[i,1] + 4*wa[i,2] - wa[i,3]) / (2*dy_a)
            mass_flux = rho_a * D_va * dw_dy_int 
            
            for j in 1:Ny_s
                T_east = (i == Nx) ? Ts_in : Ts[i+1, j]
                
                # Diffusion terms in Y (Solution)
                F_sol = (k_s * dx) / (rho_s * u_s * cp_s * dy_s^2)
                
                T_bot = (j == 1) ? Ts[i, j] : Ts[i, j-1]  # Cooled wall
                T_top = (j == Ny_s) ? Ta[i, 1] : Ts[i, j+1] # Interface with air
                
                # Latent Heat Injection (Only at interface cell j=Ny_s)
                # q_latent = m'' * H_FG
                S_latent = (j == Ny_s) ? (mass_flux * H_FG * dx) / (rho_s * u_s * cp_s * dy_s) : 0.0
                
                Ts[i,j] = (T_east + F_sol*(T_top + T_bot) + S_latent) / (1 + 2*F_sol)
            end

            # C. CONCENTRATION BALANCE (Update X)
            # m_s * dX/dx = m'' (water mass flux that dilutes the solution)
            m_s_flow = rho_s * u_s * delta_s
            X_prev = (i == Nx) ? X_in : X[i+1]
            # Water mass gained reduces desiccant fraction
            X[i] = (m_s_flow * X_prev) / (m_s_flow + mass_flux * dx)
        end

        # Stopping criterion
        if maximum(abs.(Ta - Ta_old)) < 1e-5 break end
    end

    return Ta, wa, Ts, X
end

Ta, wa, Ts, X = solve_LDAC_2D_Improved()

# --- 3. CALCULATION OF GLOBAL BALANCES ---
    
    # AIR (Outlet at i=N)
    Ta_out = sum(Ta[Nx, :]) / Ny_a
    w_out = sum(wa[Nx, :]) / Ny_a
    Ts_out = sum(Ts[1, :]) / Ny_s
    m_dot_air_dry = rho_a * u_a * delta_a
    water_transfer = m_dot_air_dry * (wa_in - w_out)
    # Moist air enthalpy: h = cp_a*T + w*(H_FG + cp_v*T)
    h_a_in = cp_a * Ta_in + wa_in * (H_FG + CP_VAPOR * Ta_in)
    h_a_out = cp_a * Ta_out + w_out * (H_FG + CP_VAPOR * Ta_out)
    @show delta_H_air = m_dot_air_dry * (h_a_in - h_a_out)

    m_water_removed = m_dot_air_dry * (wa_in - w_out)
    # SOLUTION (Outlet at i=1)
    m_dot_s_in = rho_s * u_s * delta_s
    m_dot_s_out = m_dot_s_in + m_water_removed
    
    # Solution enthalpy (Simplified: cp_s * T)
    h_s_in = cp_s * Ts_in
    h_s_out = cp_s * Ts_out
    
    delta_H_sol = (m_dot_s_out * h_s_out) - (m_dot_s_in * h_s_in)

# --- 4. BALANCE REPORT ---
println("--- MASS BALANCE ---")
println("Water removed from air: ", round(m_water_removed * 1000, digits=4), " g/s")
println("Water gained by solution: ", round(m_water_removed * 1000, digits=4), " g/s")

println("\n--- ENERGY BALANCE ---")
println("Heat released by air (Qa): ", round(delta_H_air, digits=2), " W")
println("Heat gained by solution (Qs): ", round(delta_H_sol, digits=2), " W")
error_rel = abs(delta_H_air - delta_H_sol) / delta_H_air * 100
println("Balance error: ", round(error_rel, digits=4), " %")

# --- Visualization ---
l = @layout [a ; b ; c ; d]
p1 = heatmap(1:60, (1:20).*0.25, Ta', c=:thermal, title="Air Temperature Profile (°C)", ylabel="y (mm)")
p3 = heatmap(1:60, (1:20).*0.25, wa', c=:viridis, title="Air Humidity Profile (kg/kg)", ylabel="y (mm)")
p2 = heatmap(1:60, (1:10).*0.1, Ts', c=:thermal, title="Solution Temperature Profile (°C)", ylabel="y (mm)")
p4 = plot(X, lw=3, title="Solution Concentration (X)", xlabel="X Position", color=:red, label="LiCl")
plot(p1, p2, p3, p4, layout=l, size=(800,1000))
