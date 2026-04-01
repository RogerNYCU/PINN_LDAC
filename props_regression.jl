
begin
    using Interpolations, StaticArrays
    using NonlinearSolve
    using CoolProp
    # using LinearAlgebra
    # using BenchmarkTools
end
function dot(a::T,b::T) where {T<:AbstractVector}
    tmp = zero(eltype(a))
    @inbounds @fastmath @simd for i in 1:length(a)
        tmp += a[i] * b[i]
    end
    return tmp
end

begin "i_vapor_saturation"
    const coeffs_i_sat_vapor = @SVector[-7.664856985220677e-5,0.028709375602901342,-6.289410425918317,2207.8158371199142,2.4958805266423156e6]
    function i_vapor_saturation(T::T1) where {T1}
        # T_vec = @SVector[T^4, T^3, T^2, T, 1]
        # tmp = map(*, coeffs_i_sat_vapor, T_vec)
        # return sum(tmp)
        return coeffs_i_sat_vapor[1] * T^4 + coeffs_i_sat_vapor[2] * T^3 +
         coeffs_i_sat_vapor[3] * T^2 + coeffs_i_sat_vapor[4] * T + coeffs_i_sat_vapor[5]
    end
end

# @code_warntype i_vapor_saturation(298.0)

begin "i_fg_saturation"
    const coeffs_i_fg = @SVector[-0.013914715127250973, 12.197352349203554, -5935.199560966805, 3.4957321029585632e6]
    function i_fg(T::T1) where {T1}
        # T_vec = @SVector[T^3, T^2, T, 1]
        # return dot(coeffs_i_fg, T_vec)
        return coeffs_i_fg[1] * T^3 + coeffs_i_fg[2] * T^2 + coeffs_i_fg[3] * T + coeffs_i_fg[4]
    end
end

begin "Properties Interpolations and Extrapolations"
    const Tⁿᵒᵈᵉˢ = @SVector[x + 273.15 for x in [25.0,35.0,60.0,80.0]]
    const ξⁿᵒᵈᵉˢ_2 = @SVector[x * 0.01 for x in [0.0, 50.0, 70.0, 80.0, 85.0, 90.0, 95.0]]
    const nodes = (Tⁿᵒᵈᵉˢ, ξⁿᵒᵈᵉˢ_2)
    const Tⁿᵒᵈᵉˢ_s = @SVector[x + 273.15 for x in [25.0,35.0,60.0]]
    const nodes_s = (Tⁿᵒᵈᵉˢ_s, ξⁿᵒᵈᵉˢ_2)
    const i_data = @SMatrix[
        0.0  -58000.0  -75000.0  -74000.0  -68000.0  -55000.0  -34000.0
        0.0  -57000.0  -72000.0  -72000.0  -67000.0  -54000.0  -33000.0
        0.0  -52000.0  -67000.0  -67000.0  -62000.0  -51000.0  -31000.0
        0.0  -48000.0  -62000.0  -64000.0  -59000.0  -49000.0  -30000.0
        ]
    const σ_data = @SMatrix[
        0.0719  0.0441  0.0399  0.0388  0.0376  0.0367  0.036
        0.0704  0.043   0.0391  0.0382  0.0371  0.0364  0.0359
        0.0662  0.0403  0.0371  0.0366  0.0357  0.0351  0.0347
        ]
    # ================== Interpolation and Extrapolation P_ν ==================
    const a0_p = 12.10 
    const a1_p = -28.01 
    const a2_p = 50.34 
    const a3_p = -24.63
    const b0_p = 1212.67 
    const b1_p = 772.37 
    const b2_p = 614.59 
    const b3_p = 493.33
    function _Pᵥₐₚₒᵣ_ₛₒₗ(T, ξ)
        A = a0_p + a1_p * ξ + a2_p * ξ^2 + a3_p * ξ^3
        B = b0_p + b1_p * ξ + b2_p * ξ^2 + b3_p * ξ^3
        return 10^(A - B / T) * 100.0
    end
    # ================== Interpolation and Extrapolation ρ ====================
    function _ρₛₒₗ(T, ξ)
        a0 = 804.28 + 1.585 * T - 0.0031 * T^2
        a1 = 1036.04 - 4.42 * T + 0.0057 * T^2
        a2 = -403.62 + 1.745 * T - 0.0021  * T^2
        return a0 + a1 * ξ + a2 * ξ^2
    end
    # ================== Interpolation and Extrapolation μ ====================
    function _μₛₒₗ(T, ξ)
        return (10 ^ (-1.315 + 6.1044 * ξ + 1.0944 * ξ^2 + 342.2 / T - 0.0163 * ξ * T)) * 1e-3
    end
    # ================== Interpolation and Extrapolation cp ====================
    
    function _cpₛₒₗ(T, ξ)
        return ((0.00476 * T - 4.01)* ξ + 4.21) * 1e3
    end
    # ================== Interpolation and Extrapolation 𝑘 ====================

    function _𝑘ₛₒₗ(T, ξ)
        return (-0.00103* T - 0.077) * ξ + 0.00109 * T + 0.267
    end
    # ================== Interpolation and Extrapolation 𝑖 ====================
    
    function _iₛₒₗ(T, ξ)
        i_interpolated = interpolate(nodes, i_data, Gridded(Linear()))
        i_extrapolated = extrapolate(i_interpolated, Line())
        return i_extrapolated(T, ξ)
    end
    # ================== Interpolation and Extrapolation σ ====================
    
    function _σₛₒₗ(T, ξ)
        σ_interpolated = interpolate(nodes_s, σ_data, Gridded(Linear()))
        σ_extrapolated = extrapolate(σ_interpolated, Line())
        return σ_extrapolated(T, ξ)
    end
    # ================== Find T given i_sol and ξ ====================
    # Function to find the root, given i_sol and ξ
    function calculate_T(iᵛₛₒₗ::T1, ξ::T1; T_lower=0.0 + 273.15, T_upper=95.0 + 273.15) where {T1}
        f(T::T1, p::Array{T1})= _iₛₒₗ(T, p[2]) - p[1]
        p = [iᵛₛₒₗ,ξ]
        T_span = [T_lower , T_upper]
        prob = IntervalNonlinearProblem(f, T_span, p)
        result = solve(prob, ITP())
        return result.u::Float64
    end
end

@show _Pᵥₐₚₒᵣ_ₛₒₗ(25.0 + 273.15, 0.7)
@show _ρₛₒₗ(25.0 + 273.15, 0.7)
@show _μₛₒₗ(60.0 + 273.15, 0.7)
@show _cpₛₒₗ(60.0 + 273.15, 0.7)
@show _σₛₒₗ(60.0 + 273.15, 0.7)
@show _𝑘ₛₒₗ(60.0 + 273.15, 0.7)
@show i= _iₛₒₗ(60.0 + 273.15, 0.7)
@show calculate_T(i, 0.7)


begin
    _μₐ(T) = CoolProp.PropsSI("V", "T", T, "P", 101325.0, "Air")
    _kₐ(T,ω) = CoolProp.HAPropsSI("K", "T", T, "P", 101325.0, "W", ω)
    _cpₐ(T,ω) = CoolProp.HAPropsSI("C", "T", T, "P", 101325.0, "W", ω)
    _ρₐ(T,ω) = 1.0 / CoolProp.HAPropsSI("V", "T", 25.0 + 273.15, "P", 101325.0, "W", ω)
    _νₐ(T,ω)  = _μₐ(T) / _ρₐ(T,ω) 
    _αₐ(T,ω)  = _kₐ(T,ω)  / (_ρₐ(T,ω)  * _cpₐ(T,ω))
end

