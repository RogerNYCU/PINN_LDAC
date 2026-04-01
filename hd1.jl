#mejor version al 28/01, buen balance de masa pero no de energia.
#erro masa 0.006% pero energia 37%

using Plots, Statistics, Printf, LinearAlgebra


# --- 1. AUXILIARY WATER FUNCTIONS (for LiCl correlations) ---
P_H2O(T) = exp(23.196 - 3816.44 / (T - 46.13))
ρ_H2O(τ) = 1000.0 * (1 - 0.00025 * (τ * 647.226 - 293.15)) # Simplified
η_H2O(θ) = 0.001 * exp(1.4 * (1/θ - 1))
cp_H2O(T) = 4180.0
λ_H2O(T) = 0.6
σ_H2O(θ) = 0.072 # N/m

struct LiCl end

# --- 2. YOUR OWN CORRELATIONS (Embedded) ---
@inline function f_Pᵥₐₚₒᵣ_ₛₒₗ(ξ,θ,::LiCl)
    ξ = max(0.0001, ξ); π₀, π₁, π₂, π₃, π₄, π₅ = 0.28, 4.30, 0.60, 0.21, 5.10, 0.49
    A = 2.0 - (1.0 + (ξ / π₀)^π₁)^π₂
    B = (1.0 + (ξ / π₃)^π₄)^π₅ - 1.0
    return A + B * θ
end

@inline function _Pᵥₐₚₒᵣ_ₛₒₗ(T, ξ,::LiCl)
    ξ = max(0.0001, ξ); π₆, π₇, π₈, π₉ = 0.362, -4.75, -0.40, 0.03
    T_c_H2O = 647.226; θ = T / T_c_H2O
    π₂₅ = 1.0 - (1.0 + (ξ / π₆)^π₇)^π₈ - π₉ * exp(- (ξ - 0.1)^2 / 0.005)
    return f_Pᵥₐₚₒᵣ_ₛₒₗ(ξ,θ,LiCl()) * P_H2O(T) * π₂₅
end

@inline function _Δh(T, ξ,::LiCl)
    ξ = clamp(ξ, 0.0001, 0.599); H₁, H₂, H₃, H₄, H₅, H₆ = 0.845, -1.965, -2.265, 0.6, 169.105, 457.850
    θ = T / 647.226; Δh_d0 = H₅ + H₆ * θ
    ξ_ = ξ / (H₄ - ξ)
    return Δh_d0 * (1 + (ξ_ / H₁) ^ H₂) ^ H₃ * 1000.0
end

@inline function _cpₛₒₗ(T, ξ,::LiCl)
    ξ = clamp(ξ, 0.0001, 0.599); A, B, C, D, E = 1.4398, -1.24317, -0.1207, 0.12825, 0.62934
    f1 = (ξ ≤ 0.31) ? (A*ξ + B*ξ^2 + C*ξ^3) : (D + E*ξ)
    θ = (T / 228.0) - 1.0; f2 = 58.5225*θ^0.02 - 105.6343*θ^0.04 + 47.7948*θ^0.06
    return cp_H2O(T) * (1.0 - f1 * f2)
end

@inline function _iₛₒₗ(T, ξ,::LiCl)
    return _cpₛₒₗ(T, ξ, LiCl()) * (T - 273.15) - _Δh(T, ξ, LiCl())
end


# --- 2. PARÁMETROS DEL SISTEMA ---
L, Ha, Hd = 1.0, 0.015, 0.0005 
Nx, Ny = 60, 90
dx = L/Nx
dya, dyd = Ha/Ny, Hd/Ny

# Propiedades del aire y líquido (Valores típicos)
mu_a = 1.8e-5; rho_a = 1.15; cp_a = 1006.0; ka = 0.026; Da = 2.6e-5
mu_d = 0.005;  rho_d = 1200.0; kd = 0.61; g = 9.81
h_fg = 2450000.0

# Condiciones de entrada
T_air_in, W_air_in = 308.15, 0.025
T_liq_in, xi_liq_in = 293.15, 0.40

# --- 2. CÁLCULO DE PERFILES ANALÍTICOS (Imagen 3 & 4) ---
# Resolución de u_int y dp/dx para flujo en contracorriente
u_avg_a, u_avg_d = 1.5, 0.2
# Simplificación de u_int para flujo laminar acoplado
u_int = 0.15 # Velocidad de arrastre en la interfaz

# Aire (Imagen 3, Eq 18/29)
dpdx = -12 * mu_a * u_avg_a / Ha^2 
ya_vals = range(0, Ha, length=Ny)
u_a = [u_int - (1/(2*mu_a)) * dpdx * (Ha^2 - y^2) for y in ya_vals]

# Líquido (Imagen 4, Eq 19)
yd_vals = range(0, Hd, length=Ny)
u_d = [(rho_d * g / (2*mu_d)) * (Hd^2 - (Hd - y)^2) for y in yd_vals]

# --- 3. SOLVER FVM ---
Ta = fill(T_air_in, Nx, Ny); Wa = fill(W_air_in, Nx, Ny)
Td = fill(T_liq_in, Nx, Ny); xi_d = fill(xi_liq_in, Nx, Ny)
mdot_d = fill(rho_d * u_avg_d * Hd, Nx + 1)

for iter in 1:3000
    Ta_old, Wa_old = copy(Ta), copy(Wa)
    
    # AIRE (i: 1 -> Nx)
    for i in 1:Nx
        for j in 1:Ny
            Win = (i == 1) ? W_air_in : Wa[i-1, j]
            Tin = (i == 1) ? T_air_in : Ta[i-1, j]
            
            # Difusión Y (j=1 es interfaz con líquido)
            if j == 1
                pv_s = _Pᵥₐₚₒᵣ_ₛₒₗ(Td[i, Ny], xi_d[i, Ny], LiCl())
                W_int = 0.622 * pv_s / (101325.0 - pv_s)
                W_s, T_s = W_int, Td[i, Ny]
            else
                W_s, T_s = Wa[i, j-1], Ta[i, j-1]
            end
  
            W_n = (j == Ny) ? Wa[i, j] : Wa[i, j+1] # Pared adiabática
            T_n = (j == Ny) ? Ta[i, j] : Ta[i, j+1]
            conv = u_a[j] / dx
            dw, dt = Da/dya^2, (ka/(rho_a*cp_a))/dya^2
            Wa[i,j] = (conv*Win + dw*(W_n + W_s)) / (conv + 2*dw)
            Ta[i,j] = (conv*Tin + dt*(T_n + T_s)) / (conv + 2*dt)
        end
    end

    # LÍQUIDO (i: Nx -> 1)
    for i in Nx:-1:1
        # Masa e Interfaz
        pv_s = _Pᵥₐₚₒᵣ_ₛₒₗ(Td[i, Ny], xi_d[i, Ny], LiCl())
        W_sat = 0.622 * pv_s / (101325.0 - pv_s)
        jm = (rho_a * Da / dya) * (Wa[i, 1] - W_sat)
        
        mdot_d[i] = mdot_d[i+1] + jm * dx
        xi_d[i, :] .= (mdot_d[i+1] * ((i == Nx) ? xi_liq_in : xi_d[i+1, Ny])) / mdot_d[i]

        # Energía Líquido
        for j in 1:Ny
            Tin_x = (i == Nx) ? T_liq_in : Td[i+1, j]
            Ts = (j == 1) ? Td[i, j] : Td[i, j-1]
            if j == Ny
                q_abs = jm * (h_fg + _Δh(Td[i, Ny], xi_d[i, Ny], LiCl()))
                # Implementación de la Eq de Interfaz (Imagen 2)
                Tn = Td[i, Ny] + (ka*(Ta[i, 1] - Td[i, Ny])/dya + q_abs) * (dyd/kd)
            else
                Tn = Td[i, j+1]
            end

            conv = (u_d[j] * rho_d * _cpₛₒₗ(Td[i,j], xi_d[i,j], LiCl())) / dx
            diff = kd / dyd^2
            Td[i, j] = (conv*Tin_x + diff*(Tn + Ts)) / (conv + 2*diff)
        end
    end

    if maximum(abs.(Wa - Wa_old)) < 1e-8 break end
end

# --- 4. CÁLCULO DE BALANCE ---
m_air_dry = rho_a * sum(u_a) * dya
Wa_out = sum(Wa[Nx, :] .* u_a) / sum(u_a)
Ta_out = sum(Ta[Nx, :] .* u_a) / sum(u_a)

dm_air = m_air_dry * (W_air_in - Wa_out)
dm_liq = mdot_d[1] - mdot_d[Nx+1]

h_a_in = cp_a*(T_air_in-273.15) + W_air_in*h_fg
h_a_out = cp_a*(Ta_out-273.15) + Wa_out*h_fg
Q_air = m_air_dry * (h_a_in - h_a_out)
Q_liq = mdot_d[1]*_iₛₒₗ(Td[1, Ny], xi_d[1, Ny], LiCl()) - mdot_d[Nx+1]*_iₛₒₗ(T_liq_in, xi_liq_in, LiCl())

@printf("--- Balance Final con Perfiles Analíticos ---\n")
println("Qair=", Q_air, " Qliq="   , Q_liq)
@printf("Error Masa:   %.6f %%\n", abs(dm_air - dm_liq)/dm_air * 100)
@printf("Error Energía: %.6f %%\n", abs(Q_air - Q_liq)/Q_air * 100)
p1 = heatmap(Ta' .- 273.15, title="Temp Aire (C)", cmap=:thermal)
p2 = heatmap(Td' .- 273.15, title="Temp Solución (C)", cmap=:thermal)
p3 = heatmap(Wa', title="Humedad Aire", cmap=:blues)
p4 = heatmap(xi_d', title="Conc. LiCl", cmap=:viridis)
plot(p1, p2, p3, p4, layout=(2,2))

