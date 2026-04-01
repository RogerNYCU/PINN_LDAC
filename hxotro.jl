
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


#mejor version al 29/01
using Plots, Statistics

# --- 1. PARÁMETROS FÍSICOS (ENTRADA) ---
Lx, h = 2.0, 0.05
rho, cp = 1000.0, 4180.0
kh, kc = 1.0, 1.1
um = 0.05
Th_in, Tc_in = 80.0, 20.0
DeltaT = Th_in - Tc_in

# --- 2. NORMALIZACIÓN ---
# Difusividades térmicas
alpha_h = kh / (rho * cp)
alpha_c = kc / (rho * cp)
k_int = (2 * kh * kc) / (kh + kc)
alpha_int = k_int / (rho * cp)

# Número de Péclet (Relación Convección/Difusión)
# Pe = (Velocidad * Longitud_característica) / Difusividad
# Usamos Pe modificado para la escala del dominio: (um * dy^2) / (alpha * dx)
Pe_h = (um * h^2) / (alpha_h * Lx)
Pe_c = (um * h^2) / (alpha_c * Lx)
Pe_int = (um * h^2) / (alpha_int * Lx)

# --- 3. MALLA ADIMENSIONAL ---
Nx, Ny = 100, 40
dy_star = 1.0 / Ny  # h normalizado es 1.0
caras_x = [0.5 * (1.0 - cos(pi * i / Nx)) for i in 0:Nx] # L normalizado es 1.0
dx_vec_star = [caras_x[i+1] - caras_x[i] for i in 1:Nx]

# Perfil de velocidad normalizado (u/um)
y_coords_star = range(dy_star/2, 1.0 - dy_star/2, length=Ny)
u_star = [6.0 * (yi) * (1.0 - yi) for yi in y_coords_star]

# --- 4. SOLVER ADIMENSIONAL (VARIABLES 0 a 1) ---
# theta = (T - Tc_in) / (Th_in - Tc_in)
theta_h = fill(1.0, Nx, Ny) # Entra a 1.0 (Th_in)
theta_c = fill(0.0, Nx, Ny) # Entra a 0.0 (Tc_in)

tol = 1e-10
max_iter = 50000
iter = 0

while iter < max_iter
    global iter += 1
    th_old = copy(theta_h)
    tc_old = copy(theta_c)

    # Canal Caliente (i: 1 -> Nx)
    for i in 1:Nx
        dx_s = dx_vec_star[i]
        for j in 1:Ny
            tw = (i == 1) ? 1.0 : theta_h[i-1, j]
            tn = (j == Ny) ? theta_h[i, j] : theta_h[i, j+1]
            ts = (j == 1) ? theta_c[i, Ny] : theta_h[i, j-1]
            
            # Coeficientes normalizados
            conv = u_star[j] / dx_s
            diff_n = (j == Ny) ? 0.0 : (1.0 / Pe_h) / dy_star^2
            diff_s = (j == 1) ? (1.0 / Pe_int) / dy_star^2 : (1.0 / Pe_h) / dy_star^2
            
            theta_h[i, j] = (conv * tw + diff_n * tn + diff_s * ts) / (conv + diff_n + diff_s)
        end
    end

    # Canal Frío (i: Nx -> 1)
    for i in Nx:-1:1
        dx_s = dx_vec_star[i]
        for j in 1:Ny
            te = (i == Nx) ? 0.0 : theta_c[i+1, j]
            tn = (j == Ny) ? theta_h[i, 1] : theta_c[i, j+1]
            ts = (j == 1) ? theta_c[i, j] : theta_c[i, j-1]
            
            conv = u_star[j] / dx_s
            diff_n = (j == Ny) ? (1.0 / Pe_int) / dy_star^2 : (1.0 / Pe_c) / dy_star^2
            diff_s = (j == 1) ? 0.0 : (1.0 / Pe_c) / dy_star^2
            
            theta_c[i, j] = (conv * te + diff_n * tn + diff_s * ts) / (conv + diff_n + diff_s)
        end
    end

    if max(maximum(abs.(theta_h - th_old)), maximum(abs.(theta_c - tc_old))) < tol
        break
    end
end

# --- 5. DESNORMALIZACIÓN Y BALANCE ---
# T = theta * (Th_in - Tc_in) + Tc_in
Th_final = theta_h .* DeltaT .+ Tc_in
Tc_final = theta_c .* DeltaT .+ Tc_in

# Mdot físico para el balance
m_dot = rho * sum(u_star .* um) * (h / Ny) 

# Temperaturas de mezcla (Bulk) desnormalizadas
Th_out = sum(Th_final[Nx, :] .* u_star) / sum(u_star)
Tc_out = sum(Tc_final[1, :] .* u_star) / sum(u_star)

Q_hot = m_dot * cp * (Th_in - Th_out)
Q_cold = m_dot * cp * (Tc_out - Tc_in)

# --- 6. REPORTES Y GRÁFICOS ---
println("--- Resultados (Código Normalizado) ---")
println("Pe Caliente: $(round(Pe_h, digits=2)) | Pe Frío: $(round(Pe_c, digits=2))")
println("Iteraciones: $iter")
println("Q Cedido: $(round(Q_hot, digits=4)) W | Q Ganado: $(round(Q_cold, digits=4)) W")
println("Error Balance: $(round(abs(Q_hot - Q_cold)/Q_hot * 100, digits=6)) %")



p1 = heatmap(Th_final', title="Temp. Canal Caliente (°C)", cmap=:thermal, xlabel="x", ylabel="y")
p2 = heatmap(Tc_final', title="Temp. Canal Frío (°C)", cmap=:thermal, xlabel="x", ylabel="y")
plot(p1, p2, layout=(2,1))