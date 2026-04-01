using LinearAlgebra
using Plots

# --- 1. PROPERTIES ---
function get_cp(T)
    return 4180.0 
end

function get_k(T)
    return 0.6 # Thermal conductivity (W/m·K) approx. water
end

# --- 2. 2D FVM SOLVER ---
function solve_heat_exchanger_2D()
    # Geometric parameters
    L = 5.0          # Length (m)
    H = 0.02         # Height of each channel (m)
    Nx = 1000        # Nodes in X
    Ny = 10          # Nodes in Y (per channel)
    
    dx = L / Nx
    dy = H / Ny
    
    m_dot_h = 0.5    # kg/s
    m_dot_c = 0.4    # kg/s
    rho = 1000.0     # kg/m³
    
    # Velocities (assuming unit width 1m)
    u_h = m_dot_h / (rho * H * 1.0)
    u_c = m_dot_c / (rho * H * 1.0)

    Th_in = 80.0
    Tc_in = 20.0

    # Matrix initialization (Nx x Ny)
    Th = fill(Th_in, Nx, Ny)
    Tc = fill(Tc_in, Nx, Ny)
    
    U_interface = 1200.0 # Heat transfer coefficient at dividing wall
    
    tol = 1e-5
    error = 1.0
    iter = 0

    while error > tol && iter < 2000
        Th_old = copy(Th)
        Tc_old = copy(Tc)
        
        # --- HOT FLOW (Left to right i=1 -> Nx) ---
        for i in 1:Nx
            for j in 1:Ny
                T_west = (i == 1) ? Th_in : Th[i-1, j]
                T_up = (j == Ny) ? Th[i, j] : Th[i, j+1]
                T_down = (j == 1) ? 0.0 : Th[i, j-1]
                
                cp = get_cp(Th[i, j])
                k = get_k(Th[i, j])
                
                if j > 1
                    Th[i, j] = (rho * u_h * cp * dy * T_west + k * dx / dy * (T_up + Th[i, j-1])) / 
                               (rho * u_h * cp * dy + 2 * k * dx / dy)
                else
                    # INTERFACE (j=1): Heat transfer with cold fluid
                    Th[i, j] = (rho * u_h * cp * dy * T_west + k * dx / dy * T_up + U_interface * dx * Tc[i, Ny]) / 
                               (rho * u_h * cp * dy + k * dx / dy + U_interface * dx)
                end
            end
        end

        # --- COLD FLOW (Counter-flow: Right to left i=Nx -> 1) ---
        for i in Nx:-1:1
            for j in 1:Ny
                T_east = (i == Nx) ? Tc_in : Tc[i+1, j]
                T_down = (j == 1) ? Tc[i, j] : Tc[i, j-1]
                T_up = (j == Ny) ? 0.0 : Tc[i, j+1]
                
                cp = get_cp(Tc[i, j])
                k = get_k(Tc[i, j])
                
                if j < Ny
                    Tc[i, j] = (rho * u_c * cp * dy * T_east + k * dx / dy * (T_down + Tc[i, j+1])) / 
                               (rho * u_c * cp * dy + 2 * k * dx / dy)
                else
                    # INTERFACE (j=Ny): Heat transfer with hot fluid
                    Tc[i, j] = (rho * u_c * cp * dy * T_east + k * dx / dy * T_down + U_interface * dx * Th[i, 1]) / 
                               (rho * u_c * cp * dy + k * dx / dy + U_interface * dx)
                end
            end
        end

        error = maximum(abs.(Th - Th_old)) + maximum(abs.(Tc - Tc_old))
        iter += 1
    end

    println("2D convergence achieved in $iter iterations.")
    return Th, Tc
end

Th_res, Tc_res = solve_heat_exchanger_2D()

# Average temperature at outlet (Bulk Temperature)
println("Hot outlet temperature (average): $(round(sum(Th_res[end, :])/size(Th_res,2), digits=2)) °C")
println("Cold outlet temperature (average): $(round(sum(Tc_res[1, :])/size(Tc_res,2), digits=2)) °C")
p1 = heatmap(Th_res', title="Hot Fluid Temperature (°C)", cmap=:thermal)
p2 = heatmap(Tc_res', title="Cold Fluid Temperature (°C)", cmap=:thermal)
plot(p1, p2, layout=(2,1), size=(800,800))
