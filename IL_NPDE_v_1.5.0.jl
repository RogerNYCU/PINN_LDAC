begin
    using ModelingToolkit, DomainSets
    using Plots
    using GarishPrint
    
    using Lux
    using NeuralPDE
    using Optimization
    using OptimizationOptimJL
    using OptimizationOptimisers
    # using LineSearches
    using LuxCUDA, ComponentArrays, Random
    include("props_regression.jl")
    using JLD2
end

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
    const Prₛₒₗ = cpₛₒₗ * μₛₒₗ / 𝑘ₛₒₗ 
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
    const β_Θₐ = 1 / (Reₐ * Prₐ * ARₐ)
    const β_ωₐ = 1 / (Reₐ * Scₐ * ARₐ)
    const β_Θₛₒₗ = 1 / (Reₛₒₗ * Prₛₒₗ * ARₛₒₗ)
    const β_ξₛₒₗ = 1 / (Reₛₒₗ * Scₛₒₗ * ARₛₒₗ)
end

# Parameters, variables, and derivatives
@parameters Xꜛ Yꜛₛₒₗ Yꜛₐ;
@variables Θꜛₐ(..) ωꜛₐ(..) ;
@variables Θꜛₛₒₗ(..) ξꜛₛₒₗ(..) ;

begin "problem"
    DXꜛ = Differential(Xꜛ);
    DYꜛₐ = Differential(Yꜛₐ);
    DYꜛₛₒₗ = Differential(Yꜛₛₒₗ);
    DYꜛₐ² = Differential(Yꜛₐ)^2;
    DYꜛₛₒₗ² = Differential(Yꜛₛₒₗ)^2;


    function Uꜛₛₒₗ(Yꜛₛₒₗ)
        y = Yꜛₛₒₗ * δₛₒₗ
        uₛₒₗ = g * y * (δₛₒₗ - 0.5 * y) / νₛₒₗ
        return uₛₒₗ / Uₛₒₗ_ᵣ
    end

    function Uꜛₐ(Yꜛₐ)
        y = Yꜛₐ * δₐ
        uₐ = -uᵢₙₜ - (0.5 / μₐ) * dpdx *(δₐ ^ 2 - y ^ 2)
        return -uₐ / Uₐ_ᵣ
    end

    # @register_symbolic Uꜛₛₒₗ(Yꜛ)
    # @register_symbolic Uꜛₐ(Yꜛ)

    function _coeff_Θꜛₐ_ᵢₙₜ_2(Θ_sol)
        Tᵢₙₜ = Θ_sol * ΔTₛₒₗ_ᵣ + T_w
        return (Prₛₒₗ / Scₛₒₗ) * i_fg(Tᵢₙₜ) * ξₛₒₗ_ᵢₙ / (ΔTₛₒₗ_ᵣ * cpₛₒₗ)
    end
    @register_symbolic _coeff_Θꜛₐ_ᵢₙₜ_2(Θ)

    function ωₑ(ξ,Θ_sol)
        ξ_int = ξ * ξₛₒₗ_ᵢₙ
        Tᵢₙₜ = Θ_sol * ΔTₛₒₗ_ᵣ + T_w
        # Θꜛₐ_ᵢₙₜ = (Tᵢₙₜ - T_w) / ΔTₐ_ᵣ
        ωꜛ_int = (0.62185 * _Pᵥₐₚₒᵣ_ₛₒₗ(Tᵢₙₜ,1.0 - ξ_int) / (101325.0 - _Pᵥₐₚₒᵣ_ₛₒₗ(Tᵢₙₜ,1.0 - ξ_int))) / ωₐ_ᵢₙ
        return ωꜛ_int
    end
    @register_symbolic ωₑ(ξ,Θ_sol)

    # Equations
    eqs = [
        Uꜛₐ(Yꜛₐ) * DXꜛ(Θꜛₐ(Xꜛ,Yꜛₐ)) ~ β_Θₐ * DYꜛₐ²(Θꜛₐ(Xꜛ,Yꜛₐ)),
        Uꜛₐ(Yꜛₐ) * DXꜛ(ωꜛₐ(Xꜛ,Yꜛₐ)) ~ β_ωₐ * DYꜛₐ²(ωꜛₐ(Xꜛ,Yꜛₐ)),
        Uꜛₛₒₗ(Yꜛₛₒₗ) * DXꜛ(Θꜛₛₒₗ(Xꜛ,Yꜛₛₒₗ)) ~ β_Θₛₒₗ * DYꜛₛₒₗ²(Θꜛₛₒₗ(Xꜛ,Yꜛₛₒₗ)),
        Uꜛₛₒₗ(Yꜛₛₒₗ) * DXꜛ(ξꜛₛₒₗ(Xꜛ,Yꜛₛₒₗ)) ~ β_ξₛₒₗ * DYꜛₛₒₗ²(ξꜛₛₒₗ(Xꜛ,Yꜛₛₒₗ))
    ];

    bcs = [
        Θꜛₐ(1.0, Yꜛₐ) ~ 1.0,
        ωꜛₐ(1.0, Yꜛₐ) ~ 1.0,
        Θꜛₛₒₗ(0.0, Yꜛₛₒₗ) ~ -1.5 * Yꜛₛₒₗ^2 + 3.0 * Yꜛₛₒₗ ,
        ξꜛₛₒₗ(0.0, Yꜛₛₒₗ) ~ 1.0,

        DYꜛₐ(Θꜛₐ(Xꜛ, 0.0)) ~ 0.0,
        DYꜛₐ(ωꜛₐ(Xꜛ, 0.0)) ~ 0.0,
        Θꜛₛₒₗ(Xꜛ, 0.0) ~ 0.0,
        DYꜛₛₒₗ(ξꜛₛₒₗ(Xꜛ, 0.0)) ~ 0.0,

        DYꜛₐ(Θꜛₐ(Xꜛ, 1.0)) + coeff_Θꜛₐ_ᵢₙₜ_₁ * DYꜛₛₒₗ(Θꜛₛₒₗ(Xꜛ, 1.0)) - coeff_Θꜛₐ_ᵢₙₜ_₁ * _coeff_Θꜛₐ_ᵢₙₜ_2(Θꜛₛₒₗ(Xꜛ,1.0)) * DYꜛₛₒₗ(ξꜛₛₒₗ(Xꜛ, 1.0)) ~ 0.0,
        DYꜛₐ(ωꜛₐ(Xꜛ, 1.0)) + coeff_ωꜛₐ_ᵢₙₜ * DYꜛₛₒₗ(ξꜛₛₒₗ(Xꜛ, 1.0)) ~ 0.0,
        Θꜛₐ(Xꜛ, 1.0) ~ (ΔTₛₒₗ_ᵣ / ΔTₐ_ᵣ) * Θꜛₛₒₗ(Xꜛ, 1.0),
        ωꜛₐ(Xꜛ, 1.0) ~ ωₑ(ξꜛₛₒₗ(Xꜛ, 1.0), Θꜛₛₒₗ(Xꜛ, 1.0)),
    ];


    domains = [
        Xꜛ ∈ Interval(0.0, 1.0),
        Yꜛₐ ∈ Interval(0.0, 1.0),
        Yꜛₛₒₗ ∈ Interval(0.0, 1.0),
    ];

    @named pdesystem = PDESystem(eqs, bcs, domains,[Xꜛ, Yꜛₐ, Yꜛₛₒₗ], [Θꜛₐ(Xꜛ,Yꜛₐ), ωꜛₐ(Xꜛ,Yꜛₐ), Θꜛₛₒₗ(Xꜛ,Yꜛₛₒₗ), ξꜛₛₒₗ(Xꜛ,Yꜛₛₒₗ)]);
    # pdesystem.bcs |> pprint

    # Neural network
    dim = 2 # number of dimensions;
    N_nr = 32 # number of neurons;
    chain = [
        Lux.Chain(Dense(dim, N_nr, Lux.σ),
        Dense(N_nr, N_nr, Lux.σ),
        Dense(N_nr, N_nr, Lux.σ),
        Dense(N_nr, 1)) for _ in 1:4];
    # strategy = QuadratureTraining(; batch = 200, abstol = 1e-6, reltol = 1e-6)
    # discretization = PhysicsInformedNN(chain, strategy)

    const N_poitns = 2048;
    const bcs_points = 512;
    strategy = QuasiRandomTraining(2048);
    gpu_dev = gpu_device();



    # Seeding
    rng = Random.default_rng();
    Random.seed!(rng, 0);
    ps = Lux.setup(Random.default_rng(), chain)[1] ;
    ps = ps .|> ComponentArray |> gpu_dev
    ps = f64(ps);
    discretization = PhysicsInformedNN(chain, strategy,init_params = ps);
    # discretization = PhysicsInformedNN(chain, strategy);

    # discretization |> propertynames
    # discretization.:init_params |> pprint
    prob = discretize(pdesystem, discretization);

    sym_prob = symbolic_discretize(pdesystem, discretization);

    pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions;
    bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions;

    global iter = 0;
    const loss_history= Float64[]
    callback = function (p, l)
        global iter += 1
        if iter % 100 == 0
            push!(loss_history, l)
            println("iteration number ======>   $iter")
            println("loss: ", l)
            println("pde_losses: ", map(l_ -> l_(p.u), pde_inner_loss_functions))
            println("bcs_losses: ", map(l_ -> l_(p.u), bcs_inner_loss_functions))
            println("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        end
        return false
    end
end

# @time res = solve(prob, LBFGS(linesearch = LineSearches.BackTracking());callback=callback,  maxiters = 1000)
@time res = solve(prob, OptimizationOptimisers.Adam(0.1);callback=callback,
                        maxiters = 1_000);
prob = remake(prob, u0 = res.u);
@time res = solve(prob, OptimizationOptimisers.Adam(0.01);callback=callback,
                        maxiters = 5_000);
prob = remake(prob, u0 = res.u);
@time res = solve(prob, OptimizationOptimisers.Adam(0.001);callback=callback,
                        maxiters = 5_000);
prob = remake(prob, u0 = res.u)
@time res = solve(prob, OptimizationOptimisers.Adam(0.0001);callback=callback,
                        maxiters = 5_000);
prob = remake(prob, u0 = res.u)
@time res = solve(prob, OptimizationOptimisers.Adam(0.00001);callback=callback,
                        maxiters = 5_000);

phi = discretization.phi
x_, y_a , y_sol = [DomainSets.infimum(d.domain):0.01:DomainSets.supremum(d.domain) for d in domains]

cpu_dev = cpu_device()
minimizers_ = [res.u.depvar[sym_prob.depvars[i]] for i in 1:4]
minimizers_ = minimizers_ |> cpu_dev
@save "phi_1_5.jld2" phi minimizers_ 
@save "sym_prob_1_5.jld2" sym_prob

@load "phi_1_5.jld2" phi minimizers_ 
@load "sym_prob_1_5.jld2" sym_prob

cpu_dev = cpu_device()
x_ = 0:0.001:1.0
y_a = 0:0.001:1.0
y_sol = 0:0.001:1.0

Θ_a , ω_a = [[phi[i]([x, y], minimizers_[i]) for x in x_ for y in y_a] for i in 1:2]

Θ_a = reshape(Θ_a, length(x_), length(y_a)) |> cpu_dev .|> first
ω_a = reshape(ω_a, length(x_), length(y_a)) |> cpu_dev .|> first

heatmap(Θ_a , title = "Θ_a")
contour(Θ_a , title = "Θ_a", levels = 100)
heatmap(ω_a , title = "ω_a")
contour(ω_a , title = "ω_a", levels = 100)

Θ_sol , ξ_sol = [[phi[i]([x, y], minimizers_[i]) for x in x_ for y in y_sol] for i in 3:4]

Θ_sol = reshape(Θ_sol, length(x_), length(y_sol)) |> cpu_dev .|> first
ξ_sol = reshape(ξ_sol, length(x_), length(y_sol)) |> cpu_dev .|> first

heatmap(Θ_sol , title = "Θ_sol")
heatmap(ξ_sol , title = "ξ_sol")

plot(loss_history)

plot(Θ_a[:,1] .* ΔTₐ_ᵣ .+ T_w .- 273.15, label = "Θ_a")
plot(Θ_sol[:,end] .* ΔTₛₒₗ_ᵣ .+ T_w .- 273.15, label = "Θ_sol")

plot(ω_a[:,1] .* ωₐ_ᵢₙ, label = "ω_a")
plot(1.0 .- ξ_sol[:,end] .* ξₛₒₗ_ᵢₙ , label = "ξ_sol")

@show Θ_a[:,1] .* ΔTₐ_ᵣ .+ T_w .- 273.15 |> maximum
@show ω_a[:,1] .* ωₐ_ᵢₙ |> maximum

plot(Θ_sol[1,:])
Θ_sol[1,:]
