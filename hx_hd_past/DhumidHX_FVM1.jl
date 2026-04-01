using LinearAlgebra
using Plots

# --- 1. PROPIEDADES Y CORRELACIONES ---

# Calor de vaporización/absorción del agua (J/kg)
const H_FG = 2450e3 

# Humedad de equilibrio (Simplificación de la presión de vapor sobre LiCl)
# En un modelo real, esto depende fuertemente de la concentración (X)
function get_omega_eq(T_sol, X)
    # T_sol en °C, X es concentración (0.0 a 1.0)
    # A mayor concentración, menor humedad de equilibrio (más deshumidificación)
    # A mayor temperatura, mayor humedad de equilibrio
    return 0.0005 * T_sol * (1.1 - X) + 0.005
end

# --- 2. SOLUCIONADOR FVM PARA LDAC ---

function solve_LDAC()
    # Parámetros de diseño
    L = 2.0             # Altura de la torre (m)
    N = 200             # Celdas FVM
    dz = L / N
    Area_total = 0.5    # Área de sección transversal (m2)
    perimetro = 10.0    # Perímetro de contacto (m)
    As = perimetro * dz # Área de transferencia por celda (m2)

    # Coeficientes de transferencia (Valores típicos)
    hc = 40.0           # Transferencia de calor (W/m2K)
    hm = 0.035          # Transferencia de masa (kg/m2s)

    # Condiciones de entrada - AIRE (Fluye de z=0 a z=L, i=1 a N)
    m_dot_a = 0.6       # kg/s (Aire seco)
    Ta_in = 35.0        # °C
    w_in = 0.025        # kg_agua/kg_aire (Aire muy húmedo)
    cp_a = 1006.0       # J/kgK

    # Condiciones de entrada - SOLUCIÓN (Contraflujo: de z=L a z=0, i=N a 1)
    m_dot_s = 1.2       # kg/s
    Ts_in = 15.0        # °C (Solución fría para deshumidificar)
    X_in = 0.35         # Concentración inicial (35% LiCl)
    cp_s = 3200.0       # J/kgK

    # Inicialización de perfiles
    Ta = fill(Ta_in, N)
    w  = fill(w_in, N)
    Ts = fill(Ts_in, N)
    X  = fill(X_in, N)

    # Tolerancia y bucle iterativo
    tol = 1e-6
    error = 1.0
    iter = 0
    max_iter = 2000

    while error > tol && iter < max_iter
        Ta_old = copy(Ta)
        w_old  = copy(w)
        Ts_old = copy(Ts)
        
        # FLUJO DE AIRE (i = 1 a N)
        for i in 1:N
            T_a_prev = (i == 1) ? Ta_in : Ta[i-1]
            w_prev   = (i == 1) ? w_in  : w[i-1]
            
            # Balance de Humedad (Masa)
            # ma * (w_prev - w[i]) = hm * As * (w[i] - w_eq)
            w_eq = get_omega_eq(Ts[i], X[i])
            w[i] = (m_dot_a * w_prev + hm * As * w_eq) / (m_dot_a + hm * As)
            
            # Balance de Calor Sensible Aire
            # ma * cp_a * (Ta_prev - Ta[i]) = hc * As * (Ta[i] - Ts[i])
            Ta[i] = (m_dot_a * cp_a * T_a_prev + hc * As * Ts[i]) / (m_dot_a * cp_a + hc * As)
        end

        # FLUJO DE SOLUCIÓN (Contraflujo: i = N a 1)
        for i in N:-1:1
            T_s_inlet = (i == N) ? Ts_in : Ts[i+1]
            
            # Calor ganado por la solución = Sensible del aire + Latente (condensación)
            # Usamos los deltas de la celda i
            T_a_prev = (i == 1) ? Ta_in : Ta[i-1]
            w_prev   = (i == 1) ? w_in  : w[i-1]
            
            dq_sensible = m_dot_a * cp_a * (T_a_prev - Ta[i])
            dq_latente  = m_dot_a * (w_prev - w[i]) * H_FG
            
            # Actualización de Temperatura de la Solución
            # Cs * (Ts[i] - T_s_inlet) = dq_sensible + dq_latente
            Ts[i] = T_s_inlet + (dq_sensible + dq_latente) / (m_dot_s * cp_s)
            
            # Actualización de Concentración (Masa de desecante constante)
            # m_s_entrada * X_entrada = m_s_actual * X_actual
            agua_ganada = m_dot_a * (w_prev - w[i])
            X[i] = (m_dot_s * X_in) / (m_dot_s + agua_ganada)
        end

        error = maximum(abs.(Ta - Ta_old)) + maximum(abs.(w - w_old)) + maximum(abs.(Ts - Ts_old))
        iter += 1
    end

    return Ta, w, Ts, X, iter
end

# --- 3. EJECUCIÓN Y VISUALIZACIÓN ---

Ta, w, Ts, X, iters = solve_LDAC()

println("Simulación LDAC terminada en $iters iteraciones.")
println("-"^40)
println("AIRE:")
println("  Temp entrada: 35.0 °C  -> Salida: $(round(Ta[end], digits=2)) °C")
println("  Hum. entrada: 0.025    -> Salida: $(round(w[end], digits=4)) kg/kg")
println("SOLUCIÓN DESECANTE:")
println("  Temp entrada: 15.0 °C  -> Salida: $(round(Ts[1], digits=2)) °C")
println("  Conc. entrada: 0.35    -> Salida: $(round(X[1], digits=4))")

# Gráficas
p1 = plot(Ta, label="Temp Aire (Ta)", title="Perfiles LDAC", ylabel="Temperatura (°C)", lw=2)
plot!(p1, Ts, label="Temp Solución (Ts)", lw=2)

p2 = plot(w .* 1000, label="Humedad Aire (g/kg)", color=:green, ylabel="w (g/kg)", lw=2, xlabel="Celda (Z)")

plot(p1, p2, layout=(2,1))