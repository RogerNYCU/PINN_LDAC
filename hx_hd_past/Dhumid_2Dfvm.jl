using LinearAlgebra
using Plots

function solve_LDAC_2D_Completo()
    # --- 1. MALLA Y GEOMETRÍA ---
    L, delta_a, delta_s = 1.0, 0.005, 0.001
    Nx, Ny_a, Ny_s = 60, 20, 10
    dx, dy_a, dy_s = L/Nx, delta_a/Ny_a, delta_s/Ny_s

    # --- 2. PROPIEDADES ---
    u_a, u_s = 1.5, 0.15 # Velocidades promedio
    rho_a, cp_a, k_a, D_va = 1.2, 1006.0, 0.026, 2.5e-5
    rho_s, cp_s, k_s = 1250.0, 3100.0, 0.6
    H_FG = 2450e3 # Calor de absorción J/kg

    # --- 3. MATRICES DE ESTADO ---
    Ta = fill(35.0, Nx, Ny_a); wa = fill(0.025, Nx, Ny_a) # Aire
    Ts = fill(18.0, Nx, Ny_s); X  = fill(0.35, Nx)       # Solución (X solo varía en X)

    # Entradas
    Ta_in, wa_in = 35.0, 0.025
    Ts_in, X_in  = 18.0, 0.35
    Tw_wall = 15.0 # Pared externa enfriada

    # --- 4. SOLUCIONADOR ---
    for iter in 1:800
        Ta_old, wa_old, Ts_old = copy(Ta), copy(wa), copy(Ts)

        # A. AIRE (i: 1 -> Nx | Flujo hacia adelante)
        for i in 1:Nx
            # Humedad de equilibrio en la interfaz (j=1 del aire contacta j=Ny_s de sol)
            w_eq = 0.0005 * Ts[i, Ny_s] * (1.1 - X[i]) + 0.005 

            for j in 1:Ny_a
                # --- CALOR AIRE ---
                T_west = (i == 1) ? Ta_in : Ta[i-1, j]
                T_north = (j == Ny_a) ? Ta[i, j] : Ta[i, j+1] # Simetría/Pared
                T_south = (j == 1) ? Ts[i, Ny_s] : Ta[i, j-1] # Interfaz
                
                F_heat = (k_a * dx) / (rho_a * u_a * cp_a * dy_a^2)
                Ta[i,j] = (T_west + F_heat*(T_north + T_south)) / (1 + 2*F_heat)

                # --- HUMEDAD AIRE (Masa) ---
                w_west = (i == 1) ? wa_in : wa[i-1, j]
                w_north = (j == Ny_a) ? wa[i, j] : wa[i, j+1]
                w_south = (j == 1) ? w_eq : wa[i, j-1] # j=1 es la interfaz
                
                F_mass = (D_va * dx) / (u_a * dy_a^2)
                wa[i,j] = (w_west + F_mass*(w_north + w_south)) / (1 + 2*F_mass)
            end
        end

        # B. SOLUCIÓN (i: Nx -> 1 | Contraflujo)
        for i in Nx:-1:1
            # Flujo de masa en la interfaz (Ley de Fick) para calor latente
            # m_flux = rho_a * D_va * (dw/dy)
            dw_dy = (wa[i, 2] - wa[i, 1]) / dy_a
            mass_flux_int = rho_a * D_va * dw_dy # kg/m2s consumidos/ganados

            for j in 1:Ny_s
                T_east = (i == Nx) ? Ts_in : Ts[i+1, j]
                T_top = (j == Ny_s) ? Ta[i, 1] : Ts[i, j+1]
                T_bot = (j == 1) ? Tw_wall : Ts[i, j-1]

                F_sol = (k_s * dx) / (rho_s * u_s * cp_s * dy_s^2)
                
                # Término de fuente latente solo en la interfaz (j = Ny_s)
                S_latente = (j == Ny_s) ? (mass_flux_int * H_FG * dx) / (rho_s * u_s * cp_s * dy_s) : 0.0
                
                Ts[i,j] = (T_east + F_sol*(T_top + T_bot) + S_latente) / (1 + 2*F_sol)
            end

            # --- ACTUALIZACIÓN DE CONCENTRACIÓN (X) ---
            # Balance de masa acumulado: m_s * X_in = (m_s + delta_m) * X_actual
            # m_dot_s_local = m_dot_s + (flujo_masa * area_contacto_acumulada)
            # Simplificación por celda:
            X_prev = (i == Nx) ? X_in : X[i+1]
            m_s_flow = rho_s * u_s * delta_s # kg/s por metro de ancho
            X[i] = (m_s_flow * X_prev) / (m_s_flow + mass_flux_int * dx)
        end

        # Convergencia
        if maximum(abs.(Ta - Ta_old)) < 1e-6 && maximum(abs.(wa - wa_old)) < 1e-6 break end
    end
    return Ta, wa, Ts, X
end

Ta, wa, Ts, X = solve_LDAC_2D_Completo()

# Graficando resultados
p1 = heatmap(Ta', title="Temperatura Aire (°C)", cmap=:thermal)
p2 = heatmap(wa', title="Humedad Aire (kg/kg)", cmap=:viridis)
p3 = plot(X, title="Concentración Solución (X)", xlabel="Longitud X", legend=false, lw=3)
plot(p1, p2, p3, layout=(3,1), size=(800,900))