import DifferentialEquations as DE
import Plots
import Sundials

# --- Función de Cálculos Previos (Geometría y Coeficientes) ---
function calcular_parametros_intercambiador()
    # Datos del problema
    mh = 2.0        # kg/s (Anular - Caliente)
    mc = 1.0        # kg/s (Tubo - Frío)
    Cp = 4180.0     # J/kg·K
    kf = 0.58       # W/m·K (Conductividad fluido)
    mu = 0.0013     # Pa·s (Viscosidad agua @10°C)
    Pr = 9.4        # Número de Prandtl
    kw = 400.0      # W/m·K (Conductividad pared cobre/aluminio)
    
    # Geometría (metros)
    d₀ = 0.02       # Diámetro exterior tubo interno
    espesor = 0.0015
    di = d₀ - 2 * espesor # Diámetro interior tubo interno
    Di_ext = 0.03 - 2 * espesor # Diámetro interior tubo externo (ánulo)
    
    # 1. Lad₀ del Tubo (Frío: se calienta -> n = 0.4)
    Re_c = (4 * mc) / (pi * di * mu)
    Nu_c = 0.023 * Re_c^0.8 * Pr^0.4
    hc = (Nu_c * kf) / di

    # 2. Lado del Ánulo (Caliente: se enfría -> n = 0.3)
    # Diámetro hidráulico Dh = Di_ext - do
    Dh = Di_ext - d₀
    Re_h = (4 * mh) / (pi * (Di_ext + d₀) * mu)
    Nu_h = 0.023 * Re_h^0.8 * Pr^0.3
    hh = (Nu_h * kf) / Dh

    # 3. Cálculo de U constante (basad₀ en d₀)
    # 1/U = 1/hh + resistencia_pared + resistencia_interna_escalada
    res_pared = (d₀ * log(d₀ / di)) / (2 * kw)
    inv_U = (1 / hh) + res_pared + (d₀ / (di * hc))
    U_cte = 1 / inv_U
    
    # Perímetro de transferencia (área por unidad de longitud)
    a_perimetro = pi * d₀

    # Retornamos el vector p para la DAE
    return [mh, mc, Cp, a_perimetro, U_cte]
end

# --- Función del Sistema DAE (Ecuaciones Originales) ---
# u[1] = Th, u[2] = Tc, u[3] = U
function intercambiador2(out, du, u, p, t)
    mh, mc, Cp, a, U_cte = p
    Th, Tc, U = u
    dTh, dTc, dU = du

    # Residuos de Balances de Energía
    out[1] = - (U * a / (mh * Cp)) * (Th - Tc) - dTh
    out[2] = (U * a / (mc * Cp)) * (Th - Tc) - dTc

    # Residuo de la Ecuación Algebraica (U igual al valor calculado)
    out[3] = U - U_cte
end

# --- Ejecución ---

# 1. Preparar parámetros y condiciones iniciales
p = calcular_parametros_intercambiador()
# p contiene: [mh, mc, Cp, a_perimetro, U_cte]

u₀ = [12.0, 5.0, p[5]] # Th_in, Tc_in, U_inicial

# 2. Derivadas iniciales consistentes
dTh_in = -(u₀[3] * p[4] / (p[1] * p[3])) * (u₀[1] - u₀[2])
dTc_in = (u₀[3] * p[4] / (p[2] * p[3])) * (u₀[1] - u₀[2])
du₀ = [dTh_in, dTc_in, 0.0]

# 3. Resolver
tspan = (0.0, 50.0) 
differential_vars = [true, true, false]

prob = DE.DAEProblem(intercambiador2, du₀, u₀, tspan, p, differential_vars=differential_vars)
sol = DE.solve(prob, Sundials.IDA())

# --- Gráficos ---
Plots.plot(sol, 
    layout = (3, 1), 
    vars = [(0, 1) (0, 2); (0, 3) (0, 3)], # Temperaturas arriba, U abajo
    labels = ["Th (Caliente)" "Tc (Fría)" "U Global"],
    ylabel = ["Temp (°C)" "U (W/m²K)"],
    xlabel = "Longitud (m)")



# import DifferentialEquations as DE
# import Plots
# import Sundials

# # u[1] = Th, u[2] = Tc, u[3] = U
# function intercambiador2(out, du, u, p, t)
#     mh, Cph, mc, Cpc, a = p
#     Th, Tc, U = u
#     dTh, dTc, dU = du

#     # Residuos de las Ecuaciones Diferenciales (Balances de energía)
#     out[1] = - (u[3] * a / (mh * Cph)) * (u[1] - u[2]) - du[1]
#     out[2] = (u[3] * a / (mc * Cpc)) * (u[1] - u[2]) - du[2]

#     # Residuo de la Ecuación Algebraica: U - f(T) = 0
#     out[3] = u[3] - (500.0 + 0.1 * (u[1] + u[2])) 
# end

# # --- Parámetros y Condiciones Iniciales ---
# # Parámetros: [mh, Cph, mc, Cpc, a]
# p = [0.5, 4180.0, 0.8, 4180.0, 0.2]

# # u₀: Temperaturas iniciales [Th_in, Tc_in, U_inicial]
# u₀ = [90.0, 20.0, 511.0] 

# # du₀: Derivadas iniciales para que el residuo sea 0 al inicio
# # dTh = -(U*a/(mh*Cph))*(Th-Tc)
# # dTc = (U*a/(mc*Cpc))*(Th-Tc)
# dTh_in = -(511.0 * 0.2 / (0.5 * 4180.0)) * (90.0 - 20.0)
# dTc_in = (511.0 * 0.2 / (0.8 * 4180.0)) * (90.0 - 20.0)
# du₀ = [dTh_in, dTc_in, 0.0]

# # --- Configuración del Problema DAE ---
# tspan = (0.0, 10.0) # Tiempo o longitud
# differential_vars = [true, true, false] # Th y Tc son dif, U es algebraica

# prob = DE.DAEProblem(intercambiador2, du₀, u₀, tspan, p, differential_vars=differential_vars)

# # --- Solución ---
# # Usamos IDA() de Sundials como en tu segundo ejemplo
# sol = DE.solve(prob, Sundials.IDA(), reltol=1e-8, abstol=1e-8)

# # --- Gráficos ---
# Plots.plot(sol, 
#     layout = (3, 1), 
#     labels = ["Th (Caliente)" "Tc (Fría)" "U (Global)"],
#     xlabel = "Longitud / Tiempo",
#     title = ["Temperatura Caliente" "Temperatura Fría" "Coeficiente U"])