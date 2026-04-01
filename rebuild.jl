
include("props_regression.jl")
begin "constans"
    #= 
    Nellis, G., & Klein, S. (2008). Heat Transfer. Cambridge: Cambridge University Press.
    EXAMPLE 9.2-1: Diffusion Coefficient for Air-Water Vapor Mixtures
    Bolz, R.E. and G.L. Tuve, Handbook of Tables for Applied Engineering Science, 2nd edition,CRC Press, (1976).
    =#
    Dₐ(T) = -2.775e-6 + 4.479e-8 * T + 1.656e-10 * T^2
    #= 
    Nellis, G., & Klein, S. (2008). Heat Transfer. Cambridge: Cambridge University Press.
    Infinite Dilution Diffusion Coefficients for Liquids
    A modified form of the Tyn-Calus correlation  Eq. 9-31
    Poling, B.E., J.M. Prausnitz, and J. O' Connell, The Properties of Gases and Liquids, 5th Edition,
    McGraw-Hill, New York, (2000), ISBN 0070116822 / 9780070116825. Eq. (11-9.4)
    =#
    ρ_water(T) = CoolProp.PropsSI("D", "T", T, "P", 101325.0, "Water") 
    # https://www.engineeringtoolbox.com/water-surface-tension-d_597.html
    σ_water(T) = (-1e-05 * T^2 - 0.0121 * T + 11.655) * 0.01
    v_water(T) = 18.01528 / 1000ρ_water(T)
    const IL_MW_Base = 25.0
    IL_MW(ξ) = 18.01528 * (1 - ξ) + IL_MW_Base * ξ
    vₗ(T ,ξ) = IL_MW_Base / 1000.0_ρₛₒₗ(T ,ξ)
    # Dₗ(T ,ξ) = 9.013e-16 * (v_water(T) ^ 0.267 / vₗ(T ,ξ) ^ 0.433) * (T / μₛₒₗ(T ,ξ)) * (σₛₒₗ(T ,ξ) / σ_water(T)) ^ 0.15
    Dₗ(T,ξ) = 9.013e-16 * (v_water(T) ^ 0.267 / vₗ(T ,ξ) ^ 0.433) * (T / _μₛₒₗ(T ,ξ)) * (_σₛₒₗ(T ,ξ) / σ_water(T)) ^ 0.15

    const T_w = 15.86 + 273.15 # Evaporator wall temperature
    const ΔT_supersub = 0.0 # Subcooling temperature
    # const T_w = 40.0 + 273.15 # condenser wall temperature
    const N_fin = 48
    # const N_fin = 75
    const MR = 0.052959106 / 0.025182933
    # const MR = 0.063437 / 0.037148
    # const ṁₐ_ₜₒₜ = 0.8 * 0.018709069
    const ṁₐ_ₜₒₜ = 0.025182933
    # const ṁₐ_ₜₒₜ = 0.037148
    const ṁₐ = (ṁₐ_ₜₒₜ / N_fin) * 0.5 # mass flow rate for half of the fin space
    const ṁₛₒₗ = ṁₐ * MR
    const FD = 0.205
    const Tₛₒₗ_ᵢₙ = 22.38 + 273.15
    const ξₛₒₗ_ᵢₙ = 0.28
    const ρₛₒₗ = _ρₛₒₗ(Tₛₒₗ_ᵢₙ, 1 - ξₛₒₗ_ᵢₙ)
    const g = 9.81
    const μₛₒₗ = _μₛₒₗ(T_w, 1 - ξₛₒₗ_ᵢₙ)
    const νₛₒₗ = μₛₒₗ / ρₛₒₗ
    const δₛₒₗ = ∛(3 * ṁₛₒₗ * νₛₒₗ / (ρₛₒₗ * g * FD))
    const H = 0.132
    const FS = 0.00254
    const Uₛₒₗ_ᵣ = ṁₛₒₗ / (ρₛₒₗ * δₛₒₗ * FD)
    const ARₛₒₗ = δₛₒₗ / H
    const Reₛₒₗ = Uₛₒₗ_ᵣ * δₛₒₗ / νₛₒₗ
    const 𝑘ₛₒₗ = _𝑘ₛₒₗ(T_w, 1 - ξₛₒₗ_ᵢₙ)
    const cpₛₒₗ = _cpₛₒₗ(T_w, 1 - ξₛₒₗ_ᵢₙ)
    const Prₛₒₗ = 1.1 * cpₛₒₗ * μₛₒₗ / 𝑘ₛₒₗ 
    const Dₛₒₗ = Dₗ(0.5 * (Tₛₒₗ_ᵢₙ + T_w) ,1 - ξₛₒₗ_ᵢₙ)
    const Scₛₒₗ = 2_000.0
    # const Scₛₒₗ = νₛₒₗ / Dₛₒₗ

    const Tₐ_ᵢₙ = 28.0 + 273.15
    const ωₐ_ᵢₙ = 0.019491
    const ρₐ = _ρₐ(Tₐ_ᵢₙ, ωₐ_ᵢₙ)
    const μₐ = _μₐ(Tₐ_ᵢₙ)
    const νₐ = μₐ / ρₐ
    const 𝑘ₐ = _kₐ(Tₐ_ᵢₙ, ωₐ_ᵢₙ)
    const αₐ = _αₐ(Tₐ_ᵢₙ, ωₐ_ᵢₙ)
    const Prₐ = νₐ / αₐ
    const Scₐ = νₐ / Dₐ(Tₐ_ᵢₙ)
    const δₐ = 0.5 * FS - δₛₒₗ
    const Uₐ_ᵣ = ṁₐ / (ρₐ * δₐ * FD)
    const Reₐ = Uₐ_ᵣ * δₐ / νₐ
    const ARₐ = δₐ / H
    const uᵢₙₜ = 0.5g * δₛₒₗ^2 / νₛₒₗ
    const dpdx = -(3.0 * μₐ * uᵢₙₜ / (δₐ^2)) - (3.0 * μₐ * ṁₐ / (ρₐ * (δₐ ^ 3) * FD))
    const ΔTₐ_ᵣ = Tₐ_ᵢₙ - T_w
    const ΔTₛₒₗ_ᵣ = Tₛₒₗ_ᵢₙ - T_w
    const coeff_ωꜛₐ_ᵢₙₜ = Scₐ * ARₐ * μₛₒₗ * ξₛₒₗ_ᵢₙ / (Scₛₒₗ * ARₛₒₗ * μₐ * ωₐ_ᵢₙ)
    const coeff_Θꜛₐ_ᵢₙₜ_₁ = 𝑘ₛₒₗ * ΔTₛₒₗ_ᵣ * ARₐ / (𝑘ₐ * ΔTₐ_ᵣ * ARₛₒₗ)
    _coeff_Θꜛₐ_ᵢₙₜ_2(Tᵢₙₜ) = (Prₛₒₗ / Scₛₒₗ) * i_fg(Tᵢₙₜ) * ξₛₒₗ_ᵢₙ / (ΔTₛₒₗ_ᵣ * cpₛₒₗ)
end

# ============================================================

begin "kernel"
    function Uꜛₛₒₗ(Yꜛₛₒₗ)
        y = Yꜛₛₒₗ * δₛₒₗ
        uₛₒₗ = g * y * (δₛₒₗ - 0.5 * y) / νₛₒₗ
        return uₛₒₗ / Uₛₒₗ_ᵣ
    end

    function Uꜛₐ(Yꜛₐ)
        y = Yꜛₐ * δₐ
        uₐ = -uᵢₙₜ - (0.5 / μₐ) * dpdx *(δₐ ^ 2 - y ^ 2)
        return uₐ / Uₐ_ᵣ
    end

    const γ = 1.0
    const Δx = 0.01
    const Δy = 0.01
    const M = Int(1.0 / Δx)  # 20 nodes - Forward nodes 
    const N = Int(1.0 / Δy) - 1  # 19 nodes - Intermediate nodes
    const K1 = γ * Δx / Δy^2

    const β_Θₐ = K1 / (Reₐ * Prₐ * ARₐ)
    const β_ωₐ = K1 / (Reₐ * Scₐ * ARₐ)
    const β_Θₛₒₗ = K1 / (Reₛₒₗ * Prₛₒₗ * ARₛₒₗ)
    const β_ξₛₒₗ = K1 / (Reₛₒₗ * Scₛₒₗ * ARₛₒₗ)

    const Θꜛₛₒₗ_0 = collect(0:Δx:1.0) .* (ΔT_supersub / ΔTₛₒₗ_ᵣ)

    function TDMA!(Dl,D,Du,B,X,N)
        @inbounds @simd for i in 2:N
            ω = Dl[i] / D[i-1]
            D[i] = D[i] - ω * Du[i-1]
            B[i] = B[i] - ω * B[i-1]
        end
        @inbounds X[N] = B[N] / D[N]
        @inbounds @simd for i in N-1:-1:1
            X[i] = (B[i] - Du[i] * X[i+1]) / D[i]
        end
    end

    abstract type BCType end
    struct Dirichlet <: BCType end
    struct Neumann <: BCType end

    abstract type BCLocation end
    struct StartLoc <: BCLocation end
    struct EndLoc <: BCLocation end

    # --------------------------------------------------------------------------------------------
    function BC!(Dl,D,Du,B,bc_val,Δ,::Neumann,::StartLoc)
        @inbounds D[1] = D[1] + 4.0 * Dl[1] / 3.0
        @inbounds Du[1] = Du[1] - Dl[1] / 3.0
        @inbounds B[1] = B[1] + Dl[1] * bc_val * Δ * 2.0 / 3.0
        nothing
    end

    function BC!(Dl,D,Du,B,bc_val,Δ,N::Int,::Neumann,::EndLoc)
        @inbounds B[N] = B[N] - Du[N] * bc_val * Δ * 2.0 / 3.0
        @inbounds D[N] = D[N] + 4.0 * Du[N] / 3.0
        @inbounds Dl[N] = Dl[N] - Du[N] / 3.0
        nothing
    end

    function BC!(Dl,D,Du,B,bc_val,Δ,::Dirichlet,::StartLoc) 
        @inbounds B[1] = B[1] - Dl[1] * bc_val
        nothing
    end

    function BC!(Dl,D,Du,B,bc_val,Δ,N::Int,::Dirichlet,::EndLoc) 
        @inbounds B[N] = B[N] - Du[N] * bc_val
        nothing
    end
    # --------------------------------------------------------------------------------------------

    function BC_Numan!(X,bc_val,Δ,::Neumann,::StartLoc) 
        @inbounds X[1] = -2.0 * Δ * bc_val / 3.0 + 4 * X[2] / 3.0 - X[3] / 3.0
        nothing
    end

    function BC_Numan!(X,bc_val,Δ,N::Int,::Neumann,::EndLoc) 
        @inbounds X[N+2] = 2 * Δ * bc_val / 3.0 + 4 * X[N+1] / 3.0 - X[N] / 3.0
        nothing
    end

    function BC_Dirichlet!(X,bc_val,Δ,N::Int,::Dirichlet,::EndLoc) 
        @inbounds X[N+2] = bc_val
        nothing
    end

    function BC_Dirichlet!(X,bc_val,Δ,::Dirichlet,::StartLoc) 
        @inbounds X[1] = bc_val
        nothing
    end

end


# ============================================================
function _solve_ξ!(ξꜛₛₒₗ,ξꜛₛₒₗ_ᵢₙₜ,B,α_yₛₒₗ)
    # ξꜛₛₒₗ_ᵢₙₜ = view(X,1:M)
    ∂ξꜛₛₒₗ_∂Yꜛₛₒₗ_0 = 0.0 # ∂ξₛₒₗ / ∂Yₛₒₗ at y = 0
    for i in 1:M
        Du_ξₛₒₗ = [β_ξₛₒₗ / α_yₛₒₗ[k] for k in 1:N] # Upper diagonal of ξₛₒₗ
        Dl_ξₛₒₗ = copy(Du_ξₛₒₗ) # Lower diagonal of ξₛₒₗ
        D_ξₛₒₗ = [-1.0 - 2β_ξₛₒₗ / α_yₛₒₗ[p] for p in 1:N] # Diagonal of ξₛₒₗ
        B_ξₛₒₗ = copy(B) # Right hand side of ξₛₒₗ
        @inbounds @simd for j in 1:N
            B_ξₛₒₗ[j] = -view(ξꜛₛₒₗ,1:N+2,i)[j+1] - β_ξₛₒₗ * ((1.0 - γ) / γ) * (view(ξꜛₛₒₗ,1:N+2,i)[j+2] - 2 * view(ξꜛₛₒₗ,1:N+2,i)[j+1] + view(ξꜛₛₒₗ,1:N+2,i)[j]) / α_yₛₒₗ[j]
        end
        @inbounds BC!(Dl_ξₛₒₗ,D_ξₛₒₗ,Du_ξₛₒₗ,B_ξₛₒₗ,ξꜛₛₒₗ_ᵢₙₜ[i],Δy,N,Dirichlet(),EndLoc()) # Neumann BC at y = 1.0
        @inbounds BC!(Dl_ξₛₒₗ,D_ξₛₒₗ,Du_ξₛₒₗ,B_ξₛₒₗ,∂ξꜛₛₒₗ_∂Yꜛₛₒₗ_0,Δy,Neumann(),StartLoc()) # Neumann BC at y = 0.0
        @inbounds BC_Dirichlet!(view(ξꜛₛₒₗ,:,i+1),ξꜛₛₒₗ_ᵢₙₜ[i],Δy,N,Dirichlet(),EndLoc())
        TDMA!(Dl_ξₛₒₗ,D_ξₛₒₗ,Du_ξₛₒₗ,B_ξₛₒₗ,view(ξꜛₛₒₗ,2:N+1,i+1),N)
        @inbounds BC_Numan!(view(ξꜛₛₒₗ,:,i+1),∂ξꜛₛₒₗ_∂Yꜛₛₒₗ_0,Δy,Neumann(),StartLoc())
    end
end

function _solve_Θ_sol!(Θꜛₛₒₗ,Θꜛₛₒₗ_ᵢₙₜ,α_yₛₒₗ)

    for i in 1:M
        Du_Θₛₒₗ = [β_Θₛₒₗ / α_yₛₒₗ[k] for k in 1:N] # Upper diagonal of Θₛₒₗ
        Dl_Θₛₒₗ = copy(Du_Θₛₒₗ) # Lower diagonal of Θₛₒₗ
        D_Θₛₒₗ = [-1.0 - 2β_Θₛₒₗ / α_yₛₒₗ[p] for p in 1:N] # Diagonal of Θₛₒₗ
        B_Θₛₒₗ = copy(B) # Right hand side of Θₛₒₗ
        @inbounds @simd for j in 1:N
            B_Θₛₒₗ[j] = -view(Θꜛₛₒₗ,1:N+2,i)[j+1] - β_Θₛₒₗ * ((1.0 - γ) / γ) * (view(Θꜛₛₒₗ,1:N+2,i)[j+2] - 2 * view(Θꜛₛₒₗ,1:N+2,i)[j+1] + view(Θꜛₛₒₗ,1:N+2,i)[j]) / α_yₛₒₗ[j]
        end
        @inbounds BC!(Dl_Θₛₒₗ,D_Θₛₒₗ,Du_Θₛₒₗ,B_Θₛₒₗ,Θꜛₛₒₗ_ᵢₙₜ[i],Δy,N,Dirichlet(),EndLoc()) # Dirichlet BC at y = 1.0
        @inbounds BC!(Dl_Θₛₒₗ,D_Θₛₒₗ,Du_Θₛₒₗ,B_Θₛₒₗ,Θꜛₛₒₗ_0[i+1],Δy,Dirichlet(),StartLoc()) # Dirichlet BC at y = 0.0
        @inbounds BC_Dirichlet!(view(Θꜛₛₒₗ,:,i+1),Θꜛₛₒₗ_ᵢₙₜ[i],Δy,N,Dirichlet(),EndLoc()) # Dirichlet BC at y = 1.0
        @inbounds BC_Dirichlet!(view(Θꜛₛₒₗ,:,i+1),Θꜛₛₒₗ_0[i+1],Δy,Dirichlet(),StartLoc()) # Dirichlet BC at y = 0.0
        TDMA!(Dl_Θₛₒₗ,D_Θₛₒₗ,Du_Θₛₒₗ,B_Θₛₒₗ,view(Θꜛₛₒₗ,2:N+1,i+1),N)
    end
end

function _solve_Θ_a!(Θꜛₐ,Θꜛₐ_ᵢₙₜ,α_yₐ)
    ∂Θꜛₐ_∂Yꜛₐ = 0.0 # ∂Θₐ / ∂Yₐ at interface nodes y = 0.0 or Yꜛₐ = 0.0
    for i in 1:M
        Du_Θₐ = [β_Θₐ / α_yₐ[k] for k in 1:N] # Upper diagonal of Θₐ
        Dl_Θₐ = copy(Du_Θₐ) # Lower diagonal of Θₐ
        D_Θₐ = [-1.0 - 2β_Θₐ / α_yₐ[p] for p in 1:N] # Diagonal of Θₐ
        B_Θₐ = copy(B) # Right hand side of Θₐ
        @inbounds @simd for j in 1:N
            B_Θₐ[j] = -view(Θꜛₐ,1:N+2,i)[j+1] - β_Θₐ * ((1.0 - γ) / γ) * (view(Θꜛₐ,1:N+2,i)[j+2] - 2 * view(Θꜛₐ,1:N+2,i)[j+1] + view(Θꜛₐ,1:N+2,i)[j]) / α_yₐ[j]
        end
        @inbounds BC!(Dl_Θₐ,D_Θₐ,Du_Θₐ,B_Θₐ,Θꜛₐ_ᵢₙₜ[i],Δy,N,Dirichlet(),EndLoc()) # Dirichlet BC at y = 1.0
        @inbounds BC!(Dl_Θₐ,D_Θₐ,Du_Θₐ,B_Θₐ,∂Θꜛₐ_∂Yꜛₐ,Δy,Neumann(),StartLoc()) # Neumann BC at y = 0.0
        @inbounds BC_Dirichlet!(view(Θꜛₐ,:,i+1),Θꜛₐ_ᵢₙₜ[i],Δy,N,Dirichlet(),EndLoc()) # Dirichlet BC at y = 1.0
        TDMA!(Dl_Θₐ,D_Θₐ,Du_Θₐ,B_Θₐ,view(Θꜛₐ,2:N+1,i+1),N)
        @inbounds BC_Numan!(view(Θꜛₐ,:,i+1),∂Θꜛₐ_∂Yꜛₐ,Δy,Neumann(),StartLoc()) # Neumann BC at y = 0.0
    end
end

function _solve_ω!(ωꜛₐ,ωꜛ_int,α_yₐ)
    ∂ωꜛₐ_∂Yꜛₐ = 0.0 # ∂ωₐ / ∂Yₐ at interface nodes y = 0.0 or Yꜛₐ = 0.0
    for i in 1:M
        Du_ωₐ = [β_ωₐ / α_yₐ[k] for k in 1:N] # Upper diagonal of ωₐ
        Dl_ωₐ = copy(Du_ωₐ) # Lower diagonal of ωₐ
        D_ωₐ = [-1.0 - 2β_ωₐ / α_yₐ[p] for p in 1:N] # Diagonal of ωₐ
        B_ωₐ = copy(B) # Right hand side of ωₐ
        @inbounds @simd for j in 1:N
            B_ωₐ[j] = -view(ωꜛₐ,1:N+2,i)[j+1] - β_ωₐ * ((1.0 - γ) / γ) * (view(ωꜛₐ,1:N+2,i)[j+2] - 2 * view(ωꜛₐ,1:N+2,i)[j+1] + view(ωꜛₐ,1:N+2,i)[j]) / α_yₐ[j]
        end
        @inbounds BC!(Dl_ωₐ,D_ωₐ,Du_ωₐ,B_ωₐ,ωꜛ_int[i],Δy,N,Dirichlet(),EndLoc()) # Dirichlet BC at y = 1.0
        @inbounds BC!(Dl_ωₐ,D_ωₐ,Du_ωₐ,B_ωₐ,∂ωꜛₐ_∂Yꜛₐ,Δy,Neumann(),StartLoc()) # Neumann BC at y = 0.0
        @inbounds BC_Dirichlet!(view(ωꜛₐ,:,i+1),ωꜛ_int[i],Δy,N,Dirichlet(),EndLoc()) # Dirichlet BC at y = 1.0
        TDMA!(Dl_ωₐ,D_ωₐ,Du_ωₐ,B_ωₐ,view(ωꜛₐ,2:N+1,i+1),N)
        @inbounds BC_Numan!(view(ωꜛₐ,:,i+1),∂ωꜛₐ_∂Yꜛₐ,Δy,Neumann(),StartLoc()) # Neumann BC at y = 0.0
    end
end

function solve_domain!(X,ξꜛₛₒₗ,Θꜛₛₒₗ,Θꜛₐ,ωꜛₐ,ξ_int,T_int,Θꜛₐ_ᵢₙₜ,ωꜛ_int,B,α_yₛₒₗ,α_yₐ)
    ξꜛₛₒₗ_ᵢₙₜ = view(X,1:M)
    Θꜛₛₒₗ_ᵢₙₜ = view(X,M+1:2M)
    _solve_ξ!(ξꜛₛₒₗ,ξꜛₛₒₗ_ᵢₙₜ,B,α_yₛₒₗ)
    _solve_Θ_sol!(Θꜛₛₒₗ,Θꜛₛₒₗ_ᵢₙₜ,α_yₛₒₗ)
    # --------------------------------------------------------------------------------------------
    ξ_int .= ξꜛₛₒₗ_ᵢₙₜ .* ξₛₒₗ_ᵢₙ
    T_int .= Θꜛₛₒₗ_ᵢₙₜ .* ΔTₛₒₗ_ᵣ .+ T_w
    Θꜛₐ_ᵢₙₜ .= (T_int .- T_w) ./ ΔTₐ_ᵣ |> reverse!
    ωꜛ_int .= (0.62185 .* _Pᵥₐₚₒᵣ_ₛₒₗ.(T_int,1.0 .- ξ_int) ./ (101325.0 .- _Pᵥₐₚₒᵣ_ₛₒₗ.(T_int,1.0 .- ξ_int))) ./ ωₐ_ᵢₙ |> reverse!
    # --------------------------------------------------------------------------------------------
    _solve_Θ_a!(Θꜛₐ,Θꜛₐ_ᵢₙₜ,α_yₐ)
    _solve_ω!(ωꜛₐ,ωꜛ_int,α_yₐ)    
    # --------------------------------------------------------------------------------------------
    first_der_at_int = (x , x⁻ , x⁻⁻) -> (3x - 4x⁻ + x⁻⁻) / 2Δy
    ∂ξₛₒₗ_∂Yₛₒₗ_int = map(first_der_at_int,ξꜛₛₒₗ[N+2,2:M+1],ξꜛₛₒₗ[N+1,2:M+1],ξꜛₛₒₗ[N,2:M+1])
    ∂Θₛₒₗ_∂Yₛₒₗ_int = map(first_der_at_int,Θꜛₛₒₗ[N+2,2:M+1],Θꜛₛₒₗ[N+1,2:M+1],Θꜛₛₒₗ[N,2:M+1])
    ∂Θₐ_∂Yₐ_int = map(first_der_at_int,Θꜛₐ[N+2,2:M+1],Θꜛₐ[N+1,2:M+1],Θꜛₐ[N,2:M+1]) |> reverse
    ∂ωₐ_∂Yₐ_int = map(first_der_at_int,ωꜛₐ[N+2,2:M+1],ωꜛₐ[N+1,2:M+1],ωꜛₐ[N,2:M+1]) |> reverse
    loss1 = map((x,y) -> x + coeff_ωꜛₐ_ᵢₙₜ * y, ∂ωₐ_∂Yₐ_int[1:M],∂ξₛₒₗ_∂Yₛₒₗ_int[1:M])
    loss2 = map((x,y,T,z) -> begin
                        - coeff_Θꜛₐ_ᵢₙₜ_₁ * x + coeff_Θꜛₐ_ᵢₙₜ_₁ * _coeff_Θꜛₐ_ᵢₙₜ_2(T) * y - z 
                    end,
                    ∂Θₛₒₗ_∂Yₛₒₗ_int[1:M],∂ξₛₒₗ_∂Yₛₒₗ_int[1:M],T_int[1:M],∂Θₐ_∂Yₐ_int[1:M]) 

    half_len = Int(length(loss1) / 2) +1 
    for i in 1:length(loss1)
        λ = (i <= half_len) * i * (1.0 / half_len) + (i > half_len) * (2.0 - i / half_len)
        loss1[i] = λ * loss1[i]
    end

    loss = sum(abs2,loss1) / M  + sum(abs2,loss2) / M
    return loss
end

function _segment_solve!(X)
    ξꜛₛₒₗ = zeros(Float64,N + 2,M + 1)  # allocation of ξₛₒₗ
    Θꜛₛₒₗ = zeros(Float64,N + 2,M + 1)  # allocation of Θₛₒₗ
    Θꜛₐ = zeros(Float64,N + 2,M + 1)  # allocation of Θₐ
    ωꜛₐ = zeros(Float64,N + 2,M + 1)  # allocation of ωₐ
    ξ_int = zeros(Float64,M)
    T_int = zeros(Float64,M)
    Θꜛₐ_ᵢₙₜ = zeros(Float64,M)
    ωꜛ_int = zeros(Float64,M)

    B = zeros(Float64,N)
    α_yₐ = [Uꜛₐ(i * Δy) for i in 1:N]
    α_yₛₒₗ = [Uꜛₛₒₗ(i * Δy) for i in 1:N]

    Y = 0:Δy:1.0
    @. Θꜛₛₒₗ[:,1] = -1.5 * Y^2 + 3.0 * Y ; # Θₛₒₗ at x = 0
    # Θꜛₛₒₗ[:,1] .= 1.0; # Θₛₒₗ at x = 0
    # Θꜛₛₒₗ[1,1] = 0.0
    Θꜛₐ[:,1] .= 1.0 # Θₐ at x = 0
    ξꜛₛₒₗ[:,1] .= 1.0 # ξₛₒₗ at x = 0
    ωꜛₐ[:,1] .= 1.0 # ωₐ at x = 0

    loss = solve_domain!(X,ξꜛₛₒₗ,Θꜛₛₒₗ,Θꜛₐ,ωꜛₐ,ξ_int,T_int,Θꜛₐ_ᵢₙₜ,ωꜛ_int,B,α_yₛₒₗ,α_yₐ)
    loss
end


