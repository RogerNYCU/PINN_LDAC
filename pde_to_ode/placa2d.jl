using LinearAlgebra
using DifferentialEquations
using Plots

# 1. Configuración de la malla y física
n = 9               # Nodos internos
alpha = 0.3        # Difusividad térmica (determina la velocidad de calentamiento)
tspan = (0.0, 20.0) # Intervalo de tiempo

# 2. Condiciones de contorno
function Te_top(x,n)
    return 25/n * x +75
end
aa=Te_top.(1:n+2,n)
T_top, T_bottom = 100.0, 0.0
T_left, T_right = 75.0, 50.0
function Te_bottom(x,n)
    return 50/n * x   
end
function Te_left(x,n)
    return -75.0/n * x+75
end
function Te_right(x,n)
    return -50.0/n * x +100
end
# 3. Definición del sistema de ODEs (MOL)
function heat_eq_mol!(dT, T, p, t)
    n = Int(sqrt(length(T)))
    T_mat = reshape(T, n, n)
    dT_mat = reshape(dT, n, n)
    
    for i in 1:n
        for j in 1:n
            # Vecinos (considerando condiciones de frontera)
            up    = (i > 1) ? T_mat[i-1, j] : Te_top(j,n)
            down  = (i < n) ? T_mat[i+1, j] : Te_bottom(j,n)
            left  = (j > 1) ? T_mat[i, j-1] : Te_left(i,n)
            right = (j < n) ? T_mat[i, j+1] : Te_right(i,n)
            
            # Discretización espacial del Laplaciano
            dT_mat[i, j] = alpha * (up + down + left + right - 4*T_mat[i, j])
        end
    end
end

# 4. Condición inicial: Placa a 0°C
u0 = zeros(n * n)

# 5. Resolver el sistema
prob = ODEProblem(heat_eq_mol!, u0, tspan)
sol = solve(prob, Tsit5(), saveat=0.1)

# 6. Creación de la Animación
anim = @animate for t in 0:0.2:tspan[2]
    # Extraer solución en el tiempo t y añadir bordes
    curr_T = reshape(sol(t), n, n)
    full_plot = zeros(n+2, n+2)
    full_plot[1, :] .= Te_top.(1:n+2, n)
    full_plot[end, :] .= Te_bottom.(1:n+2, n)
    full_plot[:, 1] .= Te_left.(1:n+2, n)
    full_plot[:, end] .= Te_right.(1:n+2, n)
    full_plot[2:n+1, 2:n+1] .= curr_T
    
    heatmap(full_plot, title="Evolución t = $(round(t, digits=1))", 
            clims=(0, 100), color=:viridis, yflip=true, aspect_ratio=:equal)
end

gif(anim, "calentamiento_placa.gif", fps = 15)