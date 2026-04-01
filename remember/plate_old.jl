
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, LineSearches, DomainSets
using OptimizationOptimisers
using Random, Statistics
using Plots

# 1. Parámetros Físicos y Geometría
L1, k_acero = 0.25, 50.0   # Material 1 (Izquierda)
L2, k_bronce = 0.50, 110.0 # Material 2 (Derecha)
Ly = 0.50                  # Altura de la placa
T_izq, T_der, T_sup, T_inf = 15.0, 10.0, 10.0, 40.0

# Coordenada térmica xi
xi_int = L1 / k_acero
xi_max = xi_int + (L2 / k_bronce)

@parameters xi y
@variables u(..)
Dxxi = Differential(xi)^2
Dyy = Differential(y)^2

# Ecuación de Laplace en espacio transformado (∇²u = 0)
# Nota: Al transformar solo xi, la ecuación cambia ligeramente si k_x != k_y,
# pero para materiales isotrópicos en estado estacionario se mantiene la forma.
eq = Dxxi(u(xi, y)) + Dyy(u(xi, y)) ~ 0.0

domains = [xi ∈ Interval(0.0, xi_max), 
           y ∈ Interval(0.0, Ly)]

# Definimos bcs (PDESystem los requiere aunque usemos HBC)
bcs = [u(0,y) ~ T_izq, u(xi_max,y) ~ T_der, u(xi,0) ~ T_inf, u(xi,Ly) ~ T_sup]

@named pde_system = PDESystem(eq, bcs, domains, [xi, y], [u(xi, y)])

# 2. Ansatz de Hard Boundary Conditions 2D
function assembly_hbc_2d(phi, θ, p)
    return (xi_v, y_v) -> begin
        nn_out = phi([xi_v, y_v], θ)[1]
        
        # 1. Función base que prioriza el calor desde abajo (y=0)
        # Satisface 40 en y=0 y 10 en y=Ly, x=0, x=Lx
        term_y = 40.0 * (1 - y_v/Ly) + 10.0 * (y_v/Ly)
        
        # Ajuste para que los bordes laterales (x) también sean 10
        # multiplicamos por una máscara que fuerce 10 en los lados
        G = 10.0 + (term_y - 10.0) * (4 * (xi_v/xi_max) * (1 - xi_v/xi_max))
            
        # 2. Función de distancia (se anula en todos los bordes)
        dist = xi_v * (xi_max - xi_v) * y_v * (Ly - y_v)
        
        return G + dist * nn_out
    end
end

# 3. Red Neuronal y Discretización
chain = Lux.Chain(
    Lux.Dense(2, 32, Lux.tanh),
    Lux.Dense(32, 32, Lux.tanh),
    Lux.Dense(32, 1)
)

strategy = NeuralPDE.QuasiRandomTraining(1000)
discretization = PhysicsInformedNN(chain, strategy; custom_strategy = assembly_hbc_2d)
sym_prob = symbolic_discretize(pde_system, discretization)

# 4. Entrenamiento
prob = OptimizationProblem(OptimizationFunction((θ, p) -> begin
    sum(l -> mean(abs2, l(θ)), sym_prob.loss_functions.pde_loss_functions)
end, Optimization.AutoZygote()), sym_prob.flat_init_params)

res = solve(prob, OptimizationOptimisers.Adam(0.005); maxiters = 1000)
res = solve(OptimizationProblem(prob.f, res.u), LBFGS(); maxiters = 500)

# 5. Visualización
xi_range = 0.0:xi_max/40:xi_max
y_range = 0.0:Ly/40:Ly

# Mapeo inverso de xi -> x para la gráfica
function xi_to_x(xi_v)
    return xi_v <= xi_int ? xi_v * k_acero : L1 + (xi_v - xi_int) * k_bronce
end

# Generar datos para el heatmap
xs = [xi_to_x(xi_v) for xi_v in xi_range]
ys = collect(y_range)
T_mat = [assembly_hbc_2d(discretization.phi, res.u, nothing)(xi_v, y_v) for y_v in ys, xi_v in xi_range]

heatmap(xs, ys, T_mat, title="PINN HBC 2D: Acero-Bronce",
        xlabel="x (m)", ylabel="y (m)", colormap=:inferno)
vline!([L1], color=:white, linestyle=:dash, label="Interfaz")

