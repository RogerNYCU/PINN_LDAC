using LinearAlgebra



# Calor específico (J/kg·K)
function get_cp(T_C)
    return 4180.0 # Aproximación constante para este ejemplo
end


# --- 2. CÁLCULO DEL COEFICIENTE U ---

function calculate_U() #Th, Tc, m_dot_h, m_dot_c, D_inner, D_outer
    # Propiedades en el lado caliente
    # rho_h, mu_h, k_h, cp_h = get_density(Th), get_viscosity(Th), get_conductivity(Th), get_cp(Th)
    # Re_h = (4 * m_dot_h) / (pi * D_inner * mu_h)
    # Pr_h = (cp_h * mu_h) / k_h
    # Nu_h = 0.023 * Re_h^0.8 * Pr_h^0.4
    # h_h = Nu_h * k_h / D_inner

    # # Propiedades en el lado frío (anillo)
    # rho_c, mu_c, k_c, cp_c = get_density(Tc), get_viscosity(Tc), get_conductivity(Tc), get_cp(Tc)
    # D_equiv = D_outer - D_inner
    # Re_c = (4 * m_dot_c) / (pi * (D_outer + D_inner) * mu_c)
    # Pr_c = (cp_c * mu_c) / k_c
    # Nu_c = 0.023 * Re_c^0.8 * Pr_c^0.3
    # h_c = Nu_c * k_c / D_equiv

    # Coeficiente global (despreciando resistencia del tubo)
    return 1200.0 #1 / (1/h_h + 1/h_c)
end

# --- 3. SOLUCIONADOR FVM ---

function solve_heat_exchanger()
    # Parámetros geométricos y flujo
    L = 5.0             # Longitud (m)
    N = 4000              # Número de volúmenes de control
    dx = L / N
    D_in, D_out = 0.02, 0.03
    As = pi * D_in * dx # Área superficial por celda
    
    m_dot_h = 0.5       # kg/s
    m_dot_c = 0.4       # kg/s
    Th_in = 80.0        # °C
    Tc_in = 20.0        # °C

    # Inicialización de perfiles (lineales)
    Th = collect(range(Th_in, 60.0, length=N))
    Tc = collect(range(30.0, Tc_in, length=N))
    
    tol = 1e-5
    error = 1.0
    iter = 0

    while error > tol && iter < 1000
        Th_old = copy(Th)
        Tc_old = copy(Tc)
        U = calculate_U() #Th, Tc, m_dot_h, m_dot_c, D_in, D_out
        # 1. Fluido Caliente (fluye de 1 a N)
        for i in 1:N
            T_inlet = (i == 1) ? Th_in : Th[i-1]
            Ch = m_dot_h * get_cp(Th[i])
            # Balance: Calor que entra por flujo = Calor que sale por flujo + Calor transferido
            # Ch * (T_inlet - Th[i]) = U * As * (Th[i] - Tc[i])
            # Despejando Th[i]:
            Th[i] = (Ch * T_inlet + U * As * Tc[i]) / (Ch + U * As)
        end

        # 2. Fluido Frío (fluye de N a 1)
        for i in N:-1:1
            T_inlet = (i == N) ? Tc_in : Tc[i+1]
            Cc = m_dot_c * get_cp(Tc[i])
            # Balance: Calor que entra por flujo = Calor que sale por flujo + Calor ganado
            # Cc * (T_inlet - Tc[i]) = U * As * (Tc[i] - Th[i])  <-- Error de signo común
            # El balance correcto es: Cc * (T_inlet - Tc[i]) + U * As * (Th[i] - Tc[i]) = 0
            Tc[i] = (Cc * T_inlet + U * As * Th[i]) / (Cc + U * As)
        end
        
        error = maximum(abs.(Th - Th_old)) + maximum(abs.(Tc - Tc_old))
        iter += 1
    end

    println("Convergencia alcanzada en $iter iteraciones.")
    return Th, Tc
end

# Ejecución y resultados
Th_final, Tc_final = solve_heat_exchanger()
println("Temp. Salida Caliente: $(round(Th_final[end], digits=2)) °C")
println("Temp. Salida Fría: $(round(Tc_final[1], digits=2)) °C")
