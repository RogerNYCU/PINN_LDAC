using DifferentialEquations
using Plots

# 1. Parámetros físicos y de malla
n = 10                  # Celdas por lado (10x10 = 100 celdas/ecuaciones)
L = 1.0                 # Longitud de la placa (m)
alpha = 0.01            # Difusividad térmica (k / (rho * Cp))
dx = L / n
dy = L / n

# 2. Condiciones de contorno (Bordes)
T_top, T_bottom = 100.0, 0.0
T_left, T_right = 75.0, 50.0

# 3. Función del sistema (Balance de energía en cada Volumen de Control)
function fvm_heat_2d!(dT, T, p, t)
    # T es un vector plano, lo reshapaemos a matriz para trabajar mejor
    T_mat = reshape(T, n, n)
    dT_mat = reshape(dT, n, n)
    
    for i in 1:n        # Filas (Y)
        for j in 1:n    # Columnas (X)
            
            # FLUJO OESTE (Izquierda)
            if j == 1
                q_west = alpha * (T_left - T_mat[i,j]) / (dx/2)
            else
                q_west = alpha * (T_mat[i,j-1] - T_mat[i,j]) / dx
            end
            
            # FLUJO ESTE (Derecha)
            if j == n
                q_east = alpha * (T_right - T_mat[i,j]) / (dx/2)
            else
                q_east = alpha * (T_mat[i,j+1] - T_mat[i,j]) / dx
            end
            
            # FLUJO NORTE (Arriba)
            if i == 1
                q_north = alpha * (T_top - T_mat[i,j]) / (dy/2)
            else
                q_north = alpha * (T_mat[i-1,j] - T_mat[i,j]) / dy
            end
            
            # FLUJO SUR (Abajo)
            if i == n
                q_south = alpha * (T_bottom - T_mat[i,j]) / (dy/2)
            else
                q_south = alpha * (T_mat[i+1,j] - T_mat[i,j]) / dy
            end
            
            # Balance neto en la celda (dividido por el área para obtener dT/dt)
            dT_mat[i,j] = (q_west + q_east + q_north + q_south) / dx # simplificado para dx=dy
        end
    end
end

# 4. Condición inicial y resolución
tspan = (0.0, 10.0)
T0 = fill(20.0, n * n) # Placa inicialmente a 20°C
prob = ODEProblem(fvm_heat_2d!, T0, tspan)
sol = solve(prob, Tsit5(), saveat=0.2)

# 5. Animación de la evolución térmica
anim = @animate for t in sol.t
    T_curr = reshape(sol(t), n, n)
    
    # Crear matriz extendida para visualizar bordes
    T_full = zeros(n+2, n+2)
    T_full[2:n+1, 2:n+1] .= T_curr
    T_full[1, :] .= T_top; T_full[end, :] .= T_bottom
    T_full[:, 1] .= T_left; T_full[:, end] .= T_right
    
    heatmap(T_full, title="FVM: Tiempo t = $(round(t, digits=1))", 
            clims=(0, 100), color=:thermal, yflip=true, aspect_ratio=:equal)
end

gif(anim, "evolucion_termica_fvm.gif", fps = 10)