using CoolProp
using StaticArrays

# --- Estructura para Despacho ---
struct LiCl end

# --- Tus Funciones de Propiedades (Copiadas y corregidas) ---

@inline function P_H2O(T)
    # Presión de saturación del agua pura usando CoolProp (Pa)
    return CoolProp.PropsSI("P", "T", T, "Q", 0, "Water")
end

@inline function f_Pᵥₐₚₒᵣ_ₛₒₗ(ξ, θ, ::LiCl)
    π₀, π₁, π₂, π₃, π₄, π₅ = 0.28, 4.30, 0.60, 0.21, 5.10, 0.49
    A = 2.0 - (1.0 + (ξ / π₀)^π₁)^π₂
    B = (1.0 + (ξ / π₃)^π₄)^π₅ - 1.0
    return A + B * θ
end

@inline function _Pᵥₐₚₒᵣ_ₛₒₗ(T, ξ, ::LiCl)
    π₆, π₇, π₈, π₉ = 0.362, -4.75, -0.40, 0.03
    T_c_H2O = 647.096 # Valor del paper de Conde
    θ = T / T_c_H2O
    # Corrección del exponencial a 0.0005
    π₂₅ = 1.0 - (1.0 + (ξ / π₆)^π₇)^π₈ - π₉ * exp(-(ξ - 0.1)^2 / 0.0005)
    return f_Pᵥₐₚₒᵣ_ₛₒₗ(ξ, θ, LiCl()) * P_H2O(T) * π₂₅
end

@inline function _Δh(T, ξ, ::LiCl)
    H₁, H₂, H₃, H₄, H₅, H₆ = 0.845, -1.965, -2.265, 0.6, 169.105, 457.850
    θ = T / 647.096
    Δh_d0 = H₅ + H₆ * θ
    ξ_lim = min(ξ, 0.599)
    ξ_ = ξ_lim / (H₄ - ξ_lim)
    Δh_d = Δh_d0 * (1 + (ξ_ / H₁) ^ H₂) ^ H₃
    return Δh_d * 1e3 # Retorna en J/kg
end

# --- Funciones de Aire ---
_ρₐ(T, ω) = 1.0 / CoolProp.HAPropsSI("V", "T", T, "P", 101325.0, "W", ω)
_cpₐ(T, ω) = CoolProp.HAPropsSI("C", "T", T, "P", 101325.0, "W", ω)

# --- SCRIPT DE VALIDACIÓN ---

function validar_modelo()
    # 1. Parámetros de entrada
    T_air_K = 55.0 + 273.15
    RH_air = 0.65
    T_sol_K = 10.0 + 273.15
    ξ_sol = 0.40
    P_atm = 101325.0
    h_conv = 60.0 # W/m²K (Coeficiente de calor)

    println("--- VALIDACIÓN DE PROPIEDADES ---")
    
    # 2. Aire (Humedad Absoluta)
    ω_air = CoolProp.HAPropsSI("W", "T", T_air_K, "P", P_atm, "R", RH_air)
    ρ_air = _ρₐ(T_air_K, ω_air)
    cp_air = _cpₐ(T_air_K, ω_air)
    
    # 3. Solución (Interfase)
    p_v_sol = _Pᵥₐₚₒᵣ_ₛₒₗ(T_sol_K, ξ_sol, LiCl())
    # Calculamos ω de equilibrio para esa presión de vapor
    ω_sol = CoolProp.HAPropsSI("W", "T", T_sol_K, "P", P_atm, "P_w", p_v_sol)
    
    # 4. Coeficientes de Transferencia
    # Lewis: h_m = h / (ρ * cp)
    h_m = h_conv / (ρ_air * cp_air)
    
    # 5. Flujos de Masa y Calor
    m_flux = h_m * ρ_air * (ω_air - ω_sol) # kg/(m²·s)
    
    # Calor Latente
    h_fg = CoolProp.PropsSI("H", "T", T_sol_K, "Q", 1, "Water") - 
           CoolProp.PropsSI("H", "T", T_sol_K, "Q", 0, "Water")
    q_st = _Δh(T_sol_K, ξ_sol, LiCl())
    Q_lat = m_flux * (h_fg + q_st) # W/m²
    
    # Calor Sensible
    Q_sens = h_conv * (T_air_K - T_sol_K) # W/m²

    # --- Resultados ---
    println("ω_air:         $(round(ω_air, digits=5)) kg_w/kg_as")
    println("ω_sol (intf):  $(round(ω_sol, digits=5)) kg_w/kg_as")
    println("P_v_sol:       $(round(p_v_sol / 1000, digits=4)) kPa")
    println("---------------------------------")
    println("Masa (m²/s):   $(round(m_flux, digits=6)) kg/s")
    println("Masa (m²/h):   $(round(m_flux * 3600, digits=3)) kg/h")
    println("Q_sensible:    $(round(Q_sens / 1000, digits=3)) kW/m²")
    println("Q_latente:     $(round(Q_lat / 1000, digits=3)) kW/m²")
    println("Q_total:       $(round((Q_sens + Q_lat) / 1000, digits=3)) kW/m²")
end

validar_modelo()