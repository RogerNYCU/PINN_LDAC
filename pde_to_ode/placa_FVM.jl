using DifferentialEquations
using Plots

# 1. Parámetros
n = 20                # Número de Volúmenes de Control (CV)
H = 0.02; L = 0.5; rho = 1000.0; mu = 0.001; nu = mu / rho; dpdx = -1.0; U_in = 0.1
dy = (H/2) / n        # Altura de cada celda
y_centers = collect(dy/2 : dy : H/2-dy/2) # Centros de las celdas

# 2. Función del sistema (Balance de Momentum en CV)
function fvm_flow!(du, u, p, x)
    for i in 1:n
        # 1. Calcular flujos en las caras (Difusión Viscosa)
        # Cara inferior de la celda i
        if i == 1
            # Condición de No Deslizamiento (Pared)
            # El flujo se calcula desde el centro de la celda a la pared (distancia dy/2)
            flux_bottom = mu * (u[i] - 0.0) / (dy/2)
        else
            # Flujo entre centros de celdas (distancia dy)
            flux_bottom = mu * (u[i] - u[i-1]) / dy
        end

        # Cara superior de la celda i
        if i == n
            # Condición de Simetría (Centro del canal)
            # El gradiente es cero, por lo tanto el flujo es cero
            flux_top = 0.0
        else
            flux_top = mu * (u[i+1] - u[i]) / dy
        end

        # 2. Balance de fuerzas en el Volumen de Control
        # Fuerza de presión + Flujo Neto Viscoso
        net_viscous_force = (flux_top - flux_bottom) / dy
        pressure_force = -dpdx
        
        # Ecuación: rho * u * (du/dx) = pressure_force + net_viscous_force
        if u[i] > 1e-6
            du[i] = (1 / (rho * u[i])) * (pressure_force + net_viscous_force)
        else
            du[i] = 0.0
        end
    end
end

# 3. Resolver
u0 = fill(U_in, n)
prob = ODEProblem(fvm_flow!, u0, (0.0, L))
sol = solve(prob, Rosenbrock23())

# 4. Graficar Resultados
p = plot(title="FVM: Evolución Axial de la Velocidad",
         xlabel="Distancia axial x (m)", ylabel="Velocidad (m/s)", legend=:outerright)

for i in [1, 5, 10, 15, 20]
    if i <= n
        y_pos = round(y_centers[i], digits=4)
        plot!(p, sol.t, [u[i] for u in sol.u], label="Celda y=$y_pos")
    end
end
display(p)