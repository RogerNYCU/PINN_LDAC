
#intercambiador de calor de contraflujo fvm para hacer datos de entrenamiento para pinn

using Plots
using JLD2: save # <--- Nueva librería para exportar datos

# 1. Parámetros Geométricos y de Malla
Lx, h = 2.0, 0.05       # Largo y semi-altura del canal
Nx, Ny = 50, 200         # Nodos
dx, dy = Lx/Nx, h/Ny

# 2. Propiedades Físicas
rho, cp = 1000.0, 4180.0 
kh, kc = 1.0, 1.1        
um = 0.05               

# Perfil de velocidad parabólico (Poiseuille)
y_coords = range(-h + dy/2, h - dy/2, length=Ny)
u_profile = [1.5 * um * (1 - (yi/h)^2) for yi in y_coords]

# 3. Inicialización de Matrices de Temperatura
Th = fill(80.0, Nx, Ny) 
Tc = fill(20.0, Nx, Ny) 
Th_old = copy(Th)

# 4. Bucle de Convergencia
err = 1.0
tol = 1e-5
iter = 0

alpha_h = kh / (rho * cp)
alpha_c = kc / (rho * cp)

println("Iniciando simulación FVM...")

while err > tol && iter < 5000
    global iter += 1
    Th_old .= Th
    
    # --- CANAL CALIENTE ---
    for i in 1:Nx
        for j in 1:Ny
            T_west_h = (i == 1) ? 80.0 : Th[i-1, j]
            T_north_h = (j == Ny) ? Th[i, j] : Th[i, j+1]
            T_south_h = (j == 1) ? Tc[i, Ny] : Th[i, j-1]
            conv_h = u_profile[j] / dx
            alpha_interfaz = (2 * kh * kc / (kh + kc)) / (rho * cp)
            as_h = (j == 1) ? alpha_interfaz / dy^2 : alpha_h / dy^2
            an_h = (j == Ny) ? 0.0 : alpha_h / dy^2
            
            Th[i, j] = (conv_h * T_west_h + an_h * T_north_h + as_h * T_south_h) / (conv_h + an_h + as_h)
        end
    end

    # --- CANAL FRÍO ---
    for i in Nx:-1:1
        for j in 1:Ny
            T_east_c = (i == Nx) ? 20.0 : Tc[i+1, j]
            T_south_c = (j == 1) ? Tc[i, j] : Tc[i, j-1]
            T_north_c = (j == Ny) ? Th[i, 1] : Tc[i, j+1]
            conv_c = u_profile[j] / dx
            alpha_interfaz = (2 * kh * kc / (kh + kc)) / (rho * cp)
            an_c = (j == Ny) ? alpha_interfaz / dy^2 : alpha_c / dy^2
            as_c = (j == 1) ? 0.0 : alpha_c / dy^2
            
            Tc[i, j] = (conv_c * T_east_c + an_c * T_north_c + as_c * T_south_c) / (conv_c + an_c + as_c)
        end
    end
    
    global err = maximum(abs.(Th - Th_old))
    if iter % 500 == 0
        println("Iteración $iter - Error: $err")
    end
end

# --- 5. EXPORTACIÓN DE DATOS PARA PINN ---
# Guardamos los resultados para que el otro archivo los lea
save("datos_fvm.jld2", "Th", Th, "Tc", Tc, "Nx", Nx, "Ny", Ny, "Lx", Lx, "h", h)
println("Simulación terminada. Datos guardados en 'datos_fvm.jld2'")

# --- 6. BALANCE DE ENERGÍA (Opcional para verificar) ---
m_dot = rho * sum(u_profile) * dy
Th_out = sum(Th[Nx, :] .* u_profile) / sum(u_profile)
Tc_out = sum(Tc[1, :] .* u_profile) / sum(u_profile)
Q_hot = m_dot * cp * (80.0 - Th_out)
Q_cold = m_dot * cp * (Tc_out - 20.0)

println("--- Balance FVM ---")
println("Calor: ", round(Q_hot, digits=2), " W")
println("Error Balance: ", round(abs(Q_hot - Q_cold)/Q_hot*100, digits=4), " %")