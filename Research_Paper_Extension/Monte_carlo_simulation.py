import numpy as np
import math
from scipy import linalg
from scipy.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from scipy.optimize import lsq_linear


def give_P_matix(P):
    new_P = np.zeros((P.shape[0] - 1, P.shape[1] - 1))
    for i in range(P.shape[0] - 1):
        factor = sum(P[i][:P.shape[0] - 1])
        for j in range(P.shape[0] - 1):
            new_P[i][j] = P[i][j] / factor
    return new_P

def generate_trajectory(beta, P, a):
    P = give_P_matix(P)
    states = np.arange(len(P))
    # current_state = 1  
    # trajectory = [current_state]
    trajectory = []

    for _ in range(beta - 1):
        current_state = np.random.choice(states, p=P[current_state])
        trajectory.append(current_state)
    trajectory.append(len(P))
    return trajectory

def calculate_ELB(r, P_a, G, a=1):
    P_a = np.array(P_a)
    G = np.array(G)
    exp_term = np.exp(-r * a)
    I = np.eye(P_a.shape[0])
    inverse_matrix = np.linalg.inv(I - exp_term * P_a)
    return inverse_matrix @ (exp_term * (P_a @ G))

def single_trajectory_simulation(args):
    beta, h0, G, mu, sigma, P, a = args
    trajectory = generate_trajectory(beta, P, a)
    val1 = 0
    total1 = 0
    random_normal_number=0
    steps = beta
    for time_step in range(steps):
        random_normal_number = np.random.normal(loc=0, scale=math.sqrt(2)) + val1
        value = G[trajectory[time_step]] * h0 * math.exp(- (mu - (sigma ** 2) / 2) * a * time_step - sigma * random_normal_number)
        total1 += value
        val1 = random_normal_number
    ###############################################################################################################
    total1 = max(0, h0 - total1) * math.exp(sigma * random_normal_number + beta * (mu- (sigma**2)/2))
    ##############################################################################################################
    return total1

def give_EDB_beta_seq(beta, h0, G, mu, sigma, num_trajectory, P,a):
    """Sequential Monte Carlo simulation (no nested Pool)"""
    total = 0.0
    for _ in range(num_trajectory):
        total += single_trajectory_simulation((beta, h0, G, mu, sigma, P, a))
    return total / num_trajectory

def lambda_beta(beta, alpha_1, Q):
    Q_n_minus_1_n_minus_1 = Q[:-1, :-1]
    I_n_minus_1 = np.ones((len(Q) - 1, 1))
    exp_beta_Q = linalg.expm(beta * Q_n_minus_1_n_minus_1)
    numerator = np.dot(alpha_1, exp_beta_Q @ Q_n_minus_1_n_minus_1 @ I_n_minus_1)
    denominator = np.dot(alpha_1, exp_beta_Q @ I_n_minus_1)
    return -numerator / denominator


def compute_row(h0, start_amount, r, P, G, alpha_1, Q, mu, sigma, num_of_eqns, num_trajectory,a):
    print("computing row")
    lhs_value = h0 * (1 - calculate_ELB(r, P, G, 1)[0])
    row_values = []
    for eqn_num in range(num_of_eqns):
        edb_val = give_EDB_beta_seq(eqn_num, h0, G, mu, sigma, num_trajectory, P,a)
        h = a / 10  
        F = 0.0        
        def f(b):
            lam_val = lambda_beta(b, alpha_1, Q)[0]
            return math.exp(-r * a *eqn_num+((mu - (sigma ** 2) / 2) * eqn_num)) * lam_val

        b = a * eqn_num
        for _ in range(10):
            k1 = f(b)
            k2 = f(b + h/2)
            k3 = f(b + h/2)
            k4 = f(b + h)
            F += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            b += h
            
        row_values.append(F * edb_val)
    return (h0 - start_amount, lhs_value, row_values)   

def generate_eqn_parallel(num_of_eqns, start_amount, r, P, G, alpha_1, Q, mu, sigma, num_trajectory,a):
    lhs = [0] * num_of_eqns
    coefficients = np.zeros((num_of_eqns, num_of_eqns))
    
    # Use partial to fix parameters except h0.
    func = partial(compute_row, start_amount=start_amount, r=r, P=P, G=G,
                   alpha_1=alpha_1, Q=Q, mu=mu, sigma=sigma, num_of_eqns=num_of_eqns, num_trajectory=num_trajectory,a=a)
    
    h0_values = list(range(start_amount, start_amount + num_of_eqns))
    with Pool(cpu_count()) as pool:
        results = pool.map(func, h0_values)

    for idx, lhs_value, row_values in results:
        lhs[idx] = lhs_value
        coefficients[idx, :] = row_values

    lhs = np.array(lhs).T
    x = np.linalg.solve(coefficients, lhs)
    bounds = (0,1)
    result = lsq_linear(coefficients, lhs, bounds = bounds, method = 'trf').x
    print("Residual norm:", np.linalg.norm(coefficients @ result - lhs))
    plt.plot(result)
    plt.show()

    return coefficients, lhs, result




if __name__ == "__main__":
    num_trajectory = 10000
    r = 0.06
    num_of_eqns = 20
    start_amount = 90
    a=2
    beta = 40
    h0 = 100
    alpha_1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    G = [0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.00]  
    mu = 0.0421
    sigma = 0.0117
    Q = np.array([
        [-0.1157, 0.0396, 0.0237, 0.0054, 0.0057, 0.0044, 0.0367],
        [0.5195, -1.2726, 0.4570, 0.0689, 0.0876, 0.0275, 0.1122],
        [0.1952, 0.3982, -1.2717, 0.3801, 0.0983, 0.0356, 0.1638],
        [0.0174, 0.00008, 1.0290, -1.7539, 0.4354, 0.0749, 0.1976],
        [0.0608, 0.1349, 0.1039, 0.3188, -1.1832, 0.1036, 0.4548],
        [0.0883, 0.0135, 0.01204, 0.0453, 0.000004, -0.3273, 0.1702],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    P = np.array([
        [0.9001, 0.0241, 0.0192, 0.0059, 0.0056, 0.0047, 0.04],
        [0.3067, 0.3161, 0.1567, 0.0455, 0.0427, 0.0219, 0.11],
        [0.1629, 0.1284, 0.3649, 0.1047, 0.0608, 0.0295, 0.15],
        [0.0715, 0.0627, 0.2677, 0.2356, 0.1268, 0.0473, 0.19],
        [0.0703, 0.0554, 0.0885, 0.0901, 0.3315, 0.0586, 0.30],
        [0.0753, 0.0094, 0.0159, 0.0201, 0.0048, 0.7224, 0.15],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    

    coefficients, lhs, result = generate_eqn_parallel(num_of_eqns, start_amount, r, P, G, alpha_1, Q, mu, sigma, num_trajectory,a)
    np.save('coefficients',coefficients)
    np.save('lhs',lhs)
    print(coefficients, lhs, result)
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import BSpline
    from scipy.optimize import minimize, LinearConstraint

    # --- Inputs ---
    A = coefficients
    b = lhs

    np.save('coefficients',A)
    np.save('lhs_values',b)
    # y-grid (corresponding to columns of A)
    y_grid = np.linspace(0, 49, 20)

    # --- 1. Set up B-spline basis ---
    degree = 3
    num_internal_knots = 5
    internal_knots = np.linspace(y_grid[0], y_grid[-1], num_internal_knots)

    knots = np.concatenate((
        [y_grid[0]] * degree,
        internal_knots,
        [y_grid[-1]] * degree
    ))

    n_basis = len(knots) - degree - 1
    B = np.zeros((len(y_grid), n_basis))
    B_deriv = np.zeros((len(y_grid), n_basis))

    for i in range(n_basis):
        coeff = np.zeros(n_basis)
        coeff[i] = 1.0
        spline = BSpline(knots, coeff, degree)
        B[:, i] = spline(y_grid)
        B_deriv[:, i] = spline.derivative()(y_grid)

    # --- 2. Build problem matrices
    AB = A @ B  # System matrix

    # --- 3. Set up constraints

    # Value constraints:  0 <= f(y) = B @ c <= 1
    # Monotonicity constraints:  B_deriv @ c <= 0

    # Stack constraints together
    constraint_matrix = np.vstack([
        B,          # B @ c >= 0
        -B,         # -B @ c >= -1  (i.e., B @ c <= 1)
        B_deriv     # B_deriv @ c <= 0
    ])

    lower_bounds = np.concatenate([
        np.zeros(B.shape[0]),    # B @ c >= 0
        -np.inf * np.ones(B.shape[0]),  # -B @ c unrestricted from below
        -np.inf * np.ones(B_deriv.shape[0])  # B_deriv @ c unrestricted from below
    ])

    upper_bounds = np.concatenate([
        np.inf * np.ones(B.shape[0]),  # B @ c unrestricted from above
        np.ones(B.shape[0]),           # -B @ c <= 1
        np.zeros(B_deriv.shape[0])     # B_deriv @ c <= 0
    ])

    linear_constraint = LinearConstraint(constraint_matrix, lower_bounds, upper_bounds)

    # --- 4. Solve constrained least squares
    def objective(c):
        return np.linalg.norm(AB @ c - b)**2

    # Initial guess (least squares without constraints)
    c0 = np.linalg.lstsq(AB, b, rcond=None)[0]

    res = minimize(objective, c0, constraints=[linear_constraint], method='trust-constr')

    c_opt = res.x

    # --- 5. Recover f(y)
    f_estimated = B @ c_opt
    f_estimated = np.clip(f_estimated, 0, 1)  # safety

    # --- 6. Plot
    plt.plot(y_grid, f_estimated, label='Estimated f(y)')
    plt.xlabel('y')
    plt.ylabel('f(y)')
    plt.title('Recovered smooth, decreasing f(y)')
    plt.legend()
    plt.grid()
    plt.show()
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Assuming f_estimated and y_grid are already defined from the cubic spline LS analysis:
    # For example, they might have been produced by code like:
    # A = coefficients  # from your simulation
    # b = lhs         # from your simulation
    # y_grid = np.linspace(0, 49, 20)
    # ... and then further processing to compute f_estimated using B-spline basis and LS solution.
    #
    # Here, we use that existing f_estimated value rather than simulating one.

    # Define the logistic function model (monotonic decreasing)
    def logistic_function(y, a, b):
        return 1 / (1 + np.exp(a * (y - b)))

    # Provide an initial guess for the parameters.
    # For instance, initial guess for 'a' is 0.1 and 'b' is set to the median of y_grid.
    initial_guess = [0.1, np.median(y_grid)]

    # Fit the logistic function to the f_estimated data.
    # The bounds ensure that parameter a stays nonnegative (to maintain a decreasing shape).
    params, cov = curve_fit(
        logistic_function, 
        y_grid, 
        f_estimated, 
        p0=initial_guess, 
        bounds=([0, -np.inf], [np.inf, np.inf])
    )

    # Extract the optimal parameters.
    a_opt, b_opt = params
    print("Optimal parameters: a =", a_opt, ", b =", b_opt)

    # Generate the logistic fit curve using the optimal parameters.
    f_logistic = logistic_function(y_grid, a_opt, b_opt)

    # Plot the results: the f_estimated from the cubic spline LS solution vs. the logistic fit.
    plt.plot(y_grid, f_estimated, 'o', label='Cubic Spline f_estimated')
    plt.plot(y_grid, f_logistic, '-', label='Logistic fit')
    plt.xlabel('y')
    plt.ylabel('f(y)')
    plt.title('Monotonic Decreasing f(y): Logistic Approximation')
    plt.legend()
    plt.grid()
    plt.show()