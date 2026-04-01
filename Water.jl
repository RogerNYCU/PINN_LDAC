export iᵣₛₐₜ, iᵥ_ₛₐₜ, i_fg
export P_H2O, ρ_H2O , η_H2O, λ_H2O, σ_H2O, cp_H2O


iᵥ_ₛₐₜ(T) = 10 ^ (6.697227966814859 - 273.8702703951898 /
    (T + 642.1729733423742))

i_fg(T) = -0.012234847039565204 * T^3 +
    10.546128969902453 * T^2 -
    5397.394921600233 * T +
    3.437693122636454e6


"""
    Manuel R. Conde,
    Properties of aqueous solutions of lithium and calcium chlorides: formulations for use in air conditioning equipment design,
    International Journal of Thermal Sciences,
    Volume 43, Issue 4,
    2004,
    Pages 367-382,
    ISSN 1290-0729,
    https://doi.org/10.1016/j.ijthermalsci.2003.09.003.
    (https://www.sciencedirect.com/science/article/pii/S1290072903001625)
    Abstract: The dehydration of air, for air conditioning purposes, either for human comfort or for industrial processes, is done most of the times by making it contact a surface at a temperature below its dew point. In this process not only is it necessary to cool that surface continuously, but also the air is cooled beyond the temperature necessary to the process, thus requiring reheating after dehumidification. Although the equipment for this purpose is standard and mostly low-cost, the running costs are high and high grade energy is dissipated at very low efficiency. Alternative sorption-based processes require only low grade energy for regeneration of the sorbent materials, thus incurring lower running costs. On the other hand, sorption technology equipment is usually more expensive than standard mechanical refrigeration equipment, which is essentially due to their too small market share. This paper reports the development of calculation models for the thermophysical properties of aqueous solutions of the chlorides of lithium and calcium, particularly suited for use as desiccants in sorption-based air conditioning equipment. This development has been undertaken in order to create consistent methods suitable for use in the industrial design of liquid desiccant-based air conditioning equipment. We have reviewed sources of measured data from 1850 onwards, and propose calculation models for the following properties of those aqueous solutions: Solubility boundary, vapour pressure, density, surface tension, dynamic viscosity, thermal conductivity, specific thermal capacity and differential enthalpy of dilution.
    Keywords: Liquid desiccants; Properties; Air conditioning; Open absorption; Lithium chloride; Calcium chloride; Calculation models
"""

function P_H2O(T)
    A₀ = -7.858230
    A₁ = 1.839910
    A₂  = -11.781100
    A₃ = 22.670500
    A₄ = -15.939300
    A₅ = 1.775160
    P_c_H2O = 22064e3
    T_c_H2O = 647.226
    τ = 1 - T / T_c_H2O
    RHS = (A₀ * τ + A₁ * τ^1.5 + A₂ * τ^3 + A₃ * τ^3.5 + A₄ * τ^4 + A₅ * τ^7.5) / (1 - τ)
    P = P_c_H2O * exp(RHS)
    P
end

function ρ_H2O(τ)
    ρ_c_H2O = 317.763
    B₀ = 1.9937718430 
    B₁ = 1.0985211604
    B₂ = -0.5094492996 
    B₃ = -1.7619124270
    B₄ = -44.9005480267 
    B₅ = -723692.2618632

    return ρ_c_H2O * (1 +
        B₀ * τ^(1.0 / 3.0) +
        B₁ * τ^(2.0 / 3.0) +
        B₂ * τ^(5.0 / 3.0) +
        B₃ * τ^(16.0 / 3.0) +
        B₄ * τ^(43.0 / 3.0) +
        B₅ * τ^(110.0 / 3.0))
end

function η₀_bar_H2O(T̄)
    H₀ = 1.000
    H₁ = 0.978197
    H₂ = 0.579829 
    H₃ = -0.202354
    return T̄ ^ 0.5 / (H₀ + H₁ * T̄ ^ -1 + H₂ * T̄ ^ -2 + H₃ * T̄ ^ -3)
end

const G = [
    0.5132047   0.2151778   -0.2818107  0.1778064   -0.0417661  0.0         0.0
    0.3205656   0.7317883   -1.070786   0.4605040   0.0         -0.01578386 0.0
    0.0         1.241044    -1.263184   0.2340379   0.0         0.0         0.0
    0.0         1.476783    0.0         -0.4924179  0.1600435   0.0         -0.003629481
    -0.7782567  0.0         0.0         0.0         0.0         0.0         0.0
    0.1885447   0.0         0.0         0.0         0.0         0.0         0.0
]

function η₁_bar_H2O(T̄,ρ̄ )
    _sum = 0.0
    @inbounds for i in 1:6
        @inbounds for j in 1:7
            _sum += G[i,j] * ((1.0 / T̄) - 1.0) ^ (i - 1) * (ρ̄  - 1.0) ^ (j - 1)
        end
    end
    return exp(ρ̄  * _sum)
end

function η₂_bar_H2O(T̄,ρ̄ )
    1.0
end

function η_bar_H2O(θ)
    ρꜛ = 317.763
    T̄ = θ
    τ = 1 - θ
    ρ̄  = ρ_H2O(τ) / ρꜛ
    η₀ = η₀_bar_H2O(T̄)
    η₁ = η₁_bar_H2O(T̄,ρ̄ )
    η₂ = η₂_bar_H2O(T̄,ρ̄ )
    return η₀ * η₁ * η₂
end

function η_H2O(θ)
    ηꜛ = 55.071e-6
    return η_bar_H2O(θ) * ηꜛ
end

# ====================================================================
# http://www.iapws.org/relguide/ThCond.html
# https://github.com/jjgomera/iapws/blob/fa33a677653c06f467ce8d28ce55168b67be1f04/iapws/_iapws.py#

function λ₀_bar(T̄)
    L₀ = 2.443221e-3
    L₁ = 1.323095e-2
    L₂  = 6.770357e-3
    L₃ = -3.454586e-3
    L₄ = 4.096266e-4
    # Eq 16
    return √T̄ / (L₀ + L₁ * T̄ ^ -1 + L₂ * T̄ ^ -2 + L₃ * T̄ ^ -3 + L₄ * T̄ ^ -4)
end

const L = [
    1.60397357      -0.646013523    0.111443906     0.102997357     -0.0504123634   0.00609859258
    2.33771842      -2.78843778     1.53616167      -0.463045512    0.0832827019    -0.00719201245
    2.19650529      -4.54580785     3.55777244      -1.40944978     0.275418278     -0.0205938816
    -1.21051378     1.60812989      -0.621178141    0.0716373224    0.0             0.0
    -2.7203370      4.57586331      -3.18369245     1.1168348       -0.19268305     0.012913842 
]

function λ₁_bar(T̄,ρ̄ )
    # Eq 17
    sum_i = 0.0
    @inbounds for i in 1:5
        sum_j = 0.0
        @inbounds for j in 1:6
            sum_j += L[i,j]  * (ρ̄  - 1.0) ^ (j - 1)
        end
        sum_i += sum_j * ((1.0 / T̄) - 1.0) ^ (i - 1)
    end
    return exp(ρ̄  * sum_i)
end

function λ₂_bar(T̄,ρ̄ )
    # Figure 2. Contours in the temperature-density plane where the contribution from the critical enhancement λ2 
    # to the total thermal conductivity λ equals 5 %, 1 %, 0.5 %, and 0.1 %.
    # as can be seen in the figure, the critical enhancement is only important at high temperatures and low densities.
    return 0.0
end

function λ_bar(T̄)
    ρꜛ = 317.763
    θ = T̄
    τ = 1.0 - θ
    ρ̄  = ρ_H2O(τ) / ρꜛ
    # Eq 10
    λ₀ = λ₀_bar(T̄)
    λ₁ = λ₁_bar(T̄,ρ̄ )
    λ₂ = λ₂_bar(T̄,ρ̄ )
    return λ₀ * λ₁ + λ₂
end

function λ_H2O(T)
    λꜛ = 1.0
    T_c_H2O = 647.226
    T̄ = T / T_c_H2O
    return λ_bar(T̄) * λꜛ * 1e-3
end

# ====================================================================
function σ_H2O(θ)
    σ₀ = 235.8e-3
    b = -0.625  
    μ = 1.256
    return σ₀ * (1.0 + b * (1.0 - θ)) * (1.0 - θ) ^ μ
end 

function cp_H2O(T)
    θ = (T / 228.0) - 1.0
    T_deg_C = T - 273.15
    A = 830.54602 * (T_deg_C ≤ 0.0) + 88.7891  * (T_deg_C > 0.0)
    B = -1247.52013 * (T_deg_C ≤ 0.0) + -120.1958  * (T_deg_C > 0.0)
    C = -68.60350 * (T_deg_C ≤ 0.0) + -16.9264  * (T_deg_C > 0.0)
    D = 491.27650 * (T_deg_C ≤ 0.0) + 52.4654  * (T_deg_C > 0.0)
    E = -1.80692 * (T_deg_C ≤ 0.0) + 0.10826  * (T_deg_C > 0.0)
    F = -137.51511* (T_deg_C ≤ 0.0) + 0.46988  * (T_deg_C > 0.0)
    return (A + B * (θ ^ 0.02) + C * (θ ^ 0.04) + D * (θ ^ 0.06) + E * (θ ^ 1.8) + F * (θ ^ 8.0)) * 1e3
end



