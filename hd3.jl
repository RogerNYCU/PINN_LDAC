# Dh con M 21% y E 1%, temperatura vertical uniforfe en el liquido
using Plots
using LinearAlgebra
using StaticArrays
using BenchmarkTools

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
    π₂₅ = 1.0 - (1.0 + (ξ / π₆)^π₇)^π₈ - π₉ * exp(- (ξ - 0.1)^2 / 0.0005)
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

# --- 3. SIMULATION SETTINGS ---
L, H_a, H_d = 1.0, 0.015, 0.0005
Nx, Ny = 50, 15
dx, dy_a, dy_d = L/Nx, H_a/Ny, H_d/Ny
u_a_avg, u_d_avg = 1.5, 0.2
T_air_in, W_air_in = 308.15, 0.025
T_liq_in, xi_liq_in = 293.15, 0.40
h_fg = 2450000.0; cp_a = 1006.0; rho_a = 1.15; D_a = 2.6e-5; k_a = 0.026; k_d = 0.61; rho_d = 1200.0

T_a = fill(T_air_in, Nx, Ny); W_a = fill(W_air_in, Nx, Ny)
T_d = fill(T_liq_in, Nx, Ny); ξ_d = fill(xi_liq_in, Nx, Ny)
m_dot_d = fill(rho_d * u_d_avg * H_d, Nx + 1) # Liquid mass flow per unit width


# --- 4. SOLUTION ---
for iter in 1:10000
    Wa_old = copy(W_a)
    
    # AIR (i: 1 -> Nx)
    for i in 1:Nx
        # Interface
        pv_s = _Pᵥₐₚₒᵣ_ₛₒₗ(T_d[i, Ny], ξ_d[i, Ny], LiCl())
        W_sat = 0.622 * pv_s / (101325.0 - pv_s)
        W_a[i, 1] = W_sat
        
        j_m = (rho_a * D_a / dy_a) * (W_a[i, 2] - W_a[i, 1])
        # Released heat includes h_fg AND heat of mixing
        q_abs = j_m * (h_fg + _Δh(T_d[i, Ny], ξ_d[i, Ny], LiCl()))
        
        T_int = ( (k_d/dy_d)*T_d[i, Ny-1] + (k_a/dy_a)*T_a[i, 2] + q_abs ) / (k_d/dy_d + k_a/dy_a)
        T_a[i, 1] = T_int; T_d[i, Ny] = T_int

        for j in 2:Ny
            W_in = (i == 1) ? W_air_in : W_a[i-1, j]
            T_in = (i == 1) ? T_air_in : T_a[i-1, j]
            conv = u_a_avg / dx
            W_a[i, j] = (conv*W_in + (D_a/dy_a^2)*(W_a[i, j+1 < Ny ? j+1 : j] + W_a[i, j-1])) / (conv + 2*D_a/dy_a^2)
            T_a[i, j] = (conv*T_in + (k_a/(rho_a*cp_a*dy_a^2))*(T_a[i, j+1 < Ny ? j+1 : j] + T_a[i, j-1])) / (conv + 2*k_a/(rho_a*cp_a*dy_a^2))
        end
    end

    # LIQUID (i: Nx -> 1) - Counterflow
    for i in Nx:-1:1
        j_m = (rho_a * D_a / dy_a) * (W_a[i, 2] - W_a[i, 1])
        # Liquid gains mass in direction i (from Nx to 1)
        m_dot_d[i] = m_dot_d[i+1] + j_m * dx
        
        # Salt mass balance: m_dot_in * xi_in = m_dot_out * xi_out
        ξ_d[i, Ny] = (m_dot_d[i+1] * (i==Nx ? xi_liq_in : ξ_d[i+1, Ny-1])) / m_dot_d[i]

        for j in Ny-1:-1:1
            xi_in = (i == Nx) ? xi_liq_in : ξ_d[i+1, j]
            T_in  = (i == Nx) ? T_liq_in  : T_d[i+1, j]
            # Using enthalpy for interior balance
            ξ_d[i, j] = ( (u_d_avg/dx)*xi_in + (1.5e-9/dy_d^2)*(ξ_d[i, j+1] + (j==1 ? ξ_d[i, 1] : ξ_d[i, j-1])) ) / (u_d_avg/dx + 2*1.5e-9/dy_d^2)
            T_d[i, j] = ( (u_d_avg/dx)*T_in + (0.61/(1200*2100*dy_d^2))*(T_d[i, j+1] + (j==1 ? T_d[i, 1] : T_d[i, j-1])) ) / (u_d_avg/dx + 2*0.61/(1200*2100*dy_d^2))
        end
    end
    if norm(W_a - Wa_old) < 1e-9 break end
end

# --- 5. FINAL BALANCE ---
m_a_dry = rho_a * u_a_avg * H_a
W_out = sum(W_a[Nx, :])/Ny; T_a_out = sum(T_a[Nx, :])/Ny
xi_out = sum(ξ_d[1, :])/Ny; T_d_out = sum(T_d[1, :])/Ny

# MASS
dm_air = m_a_dry * (W_air_in - W_out)
dm_liq = m_dot_d[1] - m_dot_d[Nx] # At outlet (i=1) minus inlet (i=Nx+1)

# ENERGY (Based on your enthalpies)
h_air_in = cp_a*(T_air_in-273.15) + W_air_in*h_fg
h_air_out = cp_a*(T_a_out-273.15) + W_out*h_fg
Q_air = m_a_dry * (h_air_in - h_air_out)

Q_liq = m_dot_d[1]*_iₛₒₗ(T_d_out, xi_out, LiCl()) - m_dot_d[Nx]*_iₛₒₗ(T_liq_in, xi_liq_in, LiCl())

println("Mass Error: ", round(abs(dm_air - dm_liq)/dm_air * 100, digits=2), " %")
println("Energy Error: ", round(abs(Q_air - Q_liq)/Q_air * 100, digits=2), " %")


p1 = heatmap(T_a' .- 273.15, title="Temp Air (°C)", cmap=:thermal)
p2 = heatmap(T_d' .- 273.15, title="Temp LiCl (°C)", cmap=:thermal)
p3 = heatmap(W_a', title="Humidity Air", cmap=:Blues)
p4 = heatmap(ξ_d', title="Conc. LiCl", cmap=:viridis)
plot(p1, p2, p3, p4, layout=(2,2))

