#este codigo usa Heat transfer coefficient, si funciona
using Plots

using LinearAlgebra

# 1. Parámetros Físicos y Geometría
L, W, H_canal = 1.0, 0.5, 0.02
Nx, Ny = 80, 20
dx, dy = L/Nx, H_canal/Ny

# Propiedades del fluido (ejemplo: Agua)
rho = 1000.0        # Densidad (kg/m³)
k_f = 60.6           # Conductividad térmica (W/m·K)
cp = 4180.0         # Calor específico (J/kg·K)
alpha = k_f / (rho * cp)
um = 0.1            # Velocidad media (m/s)
U = 200.0           # Coeficiente global de transferencia (W/m²K)

# 2. Perfil de Velocidad Parabólico
h = H_canal / 2
y_coords = range(-h + dy/2, h - dy/2, length=Ny)
u_profile = [1.5 * um * (1 - (yi/h)^2) for yi in y_coords]
m_dot = rho * um * (H_canal * W)

# 3. Inicialización de Matrices (Caliente y Frío)
Th = fill(80.0, Nx, Ny) # Fluido caliente entra a 80°C (Oeste)
Tc = fill(20.0, Nx, Ny) # Fluido frío entra a 20°C (Este)

# 4. Solución Iterativa (FVM)
for iter in 1:5000
    Th_old, Tc_old = copy(Th), copy(Tc)
    
    for i in 1:Nx
        for j in 1:Ny
            # --- CANAL CALIENTE (Flujo de Oeste a Este) ---
            T_west_h = (i == 1) ? 80.0 : Th[i-1, j]
            T_north_h = (j == Ny) ? Th[i, j] : Th[i, j+1] # Temp constante/Aislado
            T_south_h = (j == 1) ? Tc[i, Ny] : Th[i, j-1] # Interfaz
            
            # Términos de transporte
            conv_h = u_profile[j] / dx
            diff_h = alpha / dy^2
            
            # Balance con Coeficiente U en la base (j=1)
            source_h = (j == 1) ? -(U * (Th[i,j] - Tc[i,Ny])) / (rho * cp * dy) : 0.0
            
            Th[i,j] = (conv_h * T_west_h + diff_h * (T_north_h + T_south_h)) / (conv_h + 2*diff_h) + (source_h / (conv_h + 2*diff_h))

            # --- CANAL FRÍO (Flujo de Este a Oeste) ---
            T_east_c = (i == Nx) ? 20.0 : Tc[i+1, j]
            T_north_c = (j == Ny) ? Th[i, 1] : Tc[i, j+1] # Interfaz
            T_south_c = (j == 1) ? Tc[i, j] : Tc[i, j-1]  # Temp constante/Aislado
            
            conv_c = u_profile[j] / dx
            diff_c = alpha / dy^2
            
            # Balance con Coeficiente U en el tope (j=Ny)
            source_c = (j == Ny) ? (U * (Th[i,1] - Tc[i,j])) / (rho * cp * dy) : 0.0
            
            Tc[i,j] = (conv_c * T_east_c + diff_c * (T_north_c + T_south_c)) / (conv_c + 2*diff_c) + (source_c / (conv_c + 2*diff_c))
        end
    end
    
    if norm(Th - Th_old) < 1e-5 break end
end

# 5. BALANCE DE ENERGÍA
# Cálculo de temperaturas de mezcla (Bulk Temperatures)
Th_out = sum(Th[Nx, :] .* u_profile) / sum(u_profile)
Tc_out = sum(Tc[1, :] .* u_profile) / sum(u_profile)

Q_hot = m_dot * cp * (80.0 - Th_out)
Q_cold = m_dot * cp * (Tc_out - 20.0)

println("--- Resultados del Balance ---")
println("Calor cedido (Fluido Caliente): ", round(Q_hot, digits=2), " W")
println("Calor ganado (Fluido Frío): ", round(Q_cold, digits=2), " W")
println("Error de Balance: ", round(abs(Q_hot - Q_cold), digits=5), " W")

# 6. Visualización

p1 = heatmap(Th', title="Canal Caliente (T_h)", cmap=:thermal, ylabel="y (m)")
p2 = heatmap(Tc', title="Canal Frío (T_c)", cmap=:thermal, xlabel="x (Nodos)", ylabel="y (m)")
plot(p1, p2, layout=(2,1))