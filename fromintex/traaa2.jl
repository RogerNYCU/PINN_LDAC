using DomainSets
count_decomp = 4
x_domain = Interval(0.0, 1.0)
y_domain = Interval(0.0, 1.0)
x_domain.left
# Corrección de infimum/supremum usando .left y .right
xs_points = collect(range(x_domain.left, x_domain.right, length = count_decomp + 1))
xs_subdomains = [(xs_points[i], xs_points[i+1]) for i in 1:count_decomp]