using Plots, Statistics



# --- 1. PHYSICAL PARAMETERS (Synchronized with PINN) ---
Lx, h = 5.0, 0.01  # Modified to match PINN
rho, cp = 1000.0, 4180.0
kh, kc = 1.0, 1.1
um = 0.05
Th_in, Tc_in = 80.0, 20.0
DeltaT = Th_in - Tc_in

# --- 2. NORMALIZATION ---
alpha_h = kh / (rho * cp)
alpha_c = kc / (rho * cp)
k_int = (2 * kh * kc) / (kh + kc)
alpha_int = k_int / (rho * cp)

# Peclet number calculated with new dimensions
Pe_h = (um * h^2) / (alpha_h * Lx)
Pe_c = (um * h^2) / (alpha_c * Lx)
Pe_int = (um * h^2) / (alpha_int * Lx)

# --- 3. DIMENSIONLESS MESH ---
Nx, Ny = 100, 40
dy_star = 1.0 / Ny  
faces_x = [0.5 * (1.0 - cos(pi * i / Nx)) for i in 0:Nx] 
dx_vec_star = [faces_x[i+1] - faces_x[i] for i in 1:Nx]

# Normalized velocity profile adjusted to 1.5 from PINN
y_coords_star = range(dy_star/2, 1.0 - dy_star/2, length=Ny)
u_star = [1.5 * (1.0 - (2.0*yi - 1.0)^2) for yi in y_coords_star] # Matches PINN

# --- 4. DIMENSIONLESS SOLVER ---
theta_h = fill(1.0, Nx, Ny) 
theta_c = fill(0.0, Nx, Ny) 

tol = 1e-10
max_iter = 50000
iter = 0

while iter < max_iter
    global iter += 1
    th_old = copy(theta_h)
    tc_old = copy(theta_c)

    # Hot Channel
    for i in 1:Nx
        dx_s = dx_vec_star[i]
        for j in 1:Ny
            tw = (i == 1) ? 1.0 : theta_h[i-1, j]
            tn = (j == Ny) ? theta_h[i, j] : theta_h[i, j+1]
            ts = (j == 1) ? theta_c[i, Ny] : theta_h[i, j-1]
            
            conv = u_star[j] / dx_s
            diff_n = (j == Ny) ? 0.0 : (1.0 / Pe_h) / dy_star^2
            diff_s = (j == 1) ? (1.0 / Pe_int) / dy_star^2 : (1.0 / Pe_h) / dy_star^2
            
            theta_h[i, j] = (conv * tw + diff_n * tn + diff_s * ts) / (conv + diff_n + diff_s)
        end
    end

    # Cold Channel
    for i in Nx:-1:1
        dx_s = dx_vec_star[i]
        for j in 1:Ny
            te = (i == Nx) ? 0.0 : theta_c[i+1, j]
            tn = (j == Ny) ? theta_h[i, 1] : theta_c[i, j+1]
            ts = (j == 1) ? theta_c[i, j] : theta_c[i, j-1]
            
            conv = u_star[j] / dx_s
            diff_n = (j == Ny) ? (1.0 / Pe_int) / dy_star^2 : (1.0 / Pe_c) / dy_star^2
            diff_s = (j == 1) ? 0.0 : (1.0 / Pe_c) / dy_star^2
            
            theta_c[i, j] = (conv * te + diff_n * tn + diff_s * ts) / (conv + diff_n + diff_s)
        end
    end

    if max(maximum(abs.(theta_h - th_old)), maximum(abs.(theta_c - tc_old))) < tol
        break
    end
end

# --- 5. DENORMALIZATION AND BALANCE ---
Th_final = theta_h .* DeltaT .+ Tc_in
Tc_final = theta_c .* DeltaT .+ Tc_in

m_dot = rho * sum(u_star .* um) * (h / Ny) 
Th_out = sum(Th_final[Nx, :] .* u_star) / sum(u_star)
Tc_out = sum(Tc_final[1, :] .* u_star) / sum(u_star)
Q_hot = m_dot * cp * (Th_in - Th_out)
Q_cold = m_dot * cp * (Tc_out - Tc_in)

println("========================================")
println("Q Hot:  $Q_hot W")
println("Q Cold: $Q_cold W")
println("Error:  $(abs(Q_hot - Q_cold)/Q_hot * 100) %")
println("========================================")

p1 = heatmap(Th_final', title="FVM Hot Channel (Validation)", cmap=:thermal)
p2 = heatmap(Tc_final', title="FVM Cold Channel (Validation)", cmap=:thermal)
plot(p1, p2, layout=(2,1))