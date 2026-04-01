using DifferentialEquations
using Plots

# 1. Parámetros (mantenemos los mismos del modelo anterior)
n = 10                # Reducimos a 10 para que la gráfica no esté saturada de líneas
H = 0.02; L = 4; rho = 1000.0; mu = 0.001; nu = mu / rho; dpdx = -0.8; U_in = 0.1
dy = (H/2) / n
u0 = fill(U_in, n)

# 2. Definición del sistema (MOL)
function developing_flow!(du, u, p, x)
    for i in 1:n
        u_prev = (i == 1) ? 0.0 : u[i-1]
        u_next = (i == n) ? u[i-1] : u[i+1] # Simetría
        d2u_dy2 = (u_next - 2u[i] + u_prev) / dy^2
        
        if u[i] > 1e-6
            du[i] = (1 / u[i]) * ((-1/rho * dpdx) + (nu * d2u_dy2))
        else
            du[i] = 0.0
        end
    end
end

# 3. Resolver
prob = ODEProblem(developing_flow!, u0, (0.0, L))
sol = solve(prob, Rosenbrock23())

# 4. Visualización según tu propuesta
# sol.t contiene los puntos en X, sol.u contiene los vectores de velocidad
p = plot(title="Evolución de la Velocidad por Capas (Y)",
         xlabel="Distancia axial x (m)", 
         ylabel="Velocidad u(x) [m/s]", 
         legend=:outerright)

# Extraemos los datos para graficar cada nodo i (cada posición Y)
for i in 1:n
    # Calculamos la altura y real para la etiqueta
    y_val = round((i-1)*dy, digits=4)
    # Graficamos la velocidad del nodo i a lo largo de todo x
    plot!(p, sol.t, [u[i] for u in sol.u], label="y = $y_val m")
end

display(p)





# 6. Visualización
# y_coords = collect(0:dy:H/2)
# p = plot(title="Evolución del Perfil de Velocidad (Canal)",
#          xlabel="Velocidad (m/s)", ylabel="Posición y (m)", legend=:bottomright)

# # Graficar perfiles en diferentes posiciones x
# for x_val in [0.0, 0.5,  1.5,2.5, 2.9, 3.5, 4.0]
#     if x_val <= L
#         velocities = vcat(0.0, sol(x_val)) # Añadir el 0 de la pared
#         plot!(p, velocities, y_coords, label="x = $x_val m")
#     end
# end
# display(p)

