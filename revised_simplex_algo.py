import numpy as np
import matplotlib.pyplot as plt

#%% Helper functions
def basic_solution(n, B_inv, b, B_idx):
    """Computes basic solution."""
    xB = B_inv @ b
    x = np.zeros((n,1))
    for k in range(len(B_idx)):
        i = B_idx[k]
        x[i] = xB[k]
    return x, xB

def bland_enter_var(n, c_bar):
    """Chooses index of the entering variable according to Bland's rule."""
    return min([k for k in range(n) if c_bar[k]<0])

def bland_exit_var(n, m, u, xB, B_idx):
    """Chooses index of the exiting variable according to Bland's rule."""
    u_pos_idx = [k for k in range(n-m) if u[k]>0]
    u_pos = u[u_pos_idx, :]
    xB_pos = xB[u_pos_idx, :]
    thetas = xB_pos/u_pos
    theta_star = min(thetas).item()
    arg_theta_star = [u_pos_idx[k] for k in range(len(u_pos_idx)) if thetas[k].item()==theta_star]
    argmin_idx = min(arg_theta_star)
    exit_idx = B_idx[argmin_idx]
    return argmin_idx, exit_idx

def dantzig_enter_var(n, c_bar):
    """Chooses index of the entering variable according to Dantzig's rule."""
    c_bar_min = min(c_bar).item()
    arg_c_bar_min = [k for k in range(n) if c_bar[k]==c_bar_min]
    return min(arg_c_bar_min)

def dantzig_exit_var(n, m, u, xB, B_idx):
    """Chooses index of the exiting variable according to Dantzig's rule."""
    # Same as Bland's rule
    return bland_exit_var(n, m, u, xB, B_idx)

def display_path(path):
    """Display the basic feasible solutions visited by the Simplex algo."""
    for k in range(len(path)):
        print(f"BFS {k+1} = {np.squeeze(path[k].T)}")
    return

def generate_problem(N, init_basis="slack", B_idx=None):
    """Generate test problem."""
    # Build the LP's matrices
    c = np.zeros((2*N,1))
    b = np.zeros((N,1))
    M = np.eye(N)
    for k in range(N):
        c[k] = -3**(N-(k+1))
        b[k] = 3**(2*(k+1)-2)
        for l in range(k+1, N):
            M[l][k] = 2*3**(l-k)
    A = np.concatenate((M,np.eye(N)), axis=1)
    # Build initial basis and check feasibility of initial basic solution
    if init_basis=="slack":
        B_idx = [N+k for k in range(N)]
    elif init_basis=="custom":
        assert type(B_idx)==list, "Specify a B_idx input other than None."
        assert len(B_idx)==N, "Specify a B_idx with N elements."
        B_idx = B_idx
    elif init_basis=="random":
        while True:
            B_idx = list(rng.choice(2*N, N, replace=False))
            B = A[:,B_idx]
            if np.linalg.det(B) != 0:
                B_inv = np.linalg.inv(B)
                x,_ = basic_solution(2*N, B_inv, b, B_idx)
                if (x>=0).all():
                    break
    return A, b, c, B_idx

#%% Simplex implementations
def revised_simplex(A, b, c, method, B_idx, error=1e-4):
    """An implementation of the revised Simplex algorithm as detailed in 
    "Introduction to Linear Optimization" by Bertsemas and Tsitsiklis (chap. 3)"""
    # Checks on input matrix dimensions
    n = c.shape[0]
    m = b.shape[0]
    assert A.shape[0]==m, "The numbers of rows of A must be equal to the number of elements of b."
    assert A.shape[1]==n, "The number of columns of A must be equal to the number of elements of c."

    # Initialize algorithm
    B = A[:, B_idx] # basis matrix
    assert np.linalg.det(B)!=0, "Chosen basis matrix not invertible. Choose another basis matrix."
    B_inv = np.linalg.inv(B) # inverse of basis matrix
    x, xB = basic_solution(n, B_inv, b, B_idx) # initial basic solution
    assert (x >= 0).all(), "Initial basic solution is infeasible. Choose another basis matrix."
    path = [x] # store the BFS visited by the algo

    while True:
        # Compute the reduced costs
        cB = np.array([c[k].item() for k in B_idx]).reshape((-1,1))
        c_bar = c - (cB.T@B_inv@A).T
        for k in range(n):
            if abs(c_bar[k]) < error:
                c_bar[k] = 0

        # Check nonnegativity of reduced costs (optimality condition)
        if (c_bar >= 0).all():
            x, _ = basic_solution(n, B_inv,b,B_idx)
            print("All reduced costs are nonnegative.")
            print(f"Optimal solution = {np.squeeze(x.T)}")
            print(f"Minimum cost = {(c.T@x).item()}")
            break
        else:
            # Choose an entering variable
            if method == "Bland":
                # Apply Bland's rule
                enter_idx = bland_enter_var(n, c_bar)
            elif method == "Dantzig":
                # Apply Dantzig's rule
                enter_idx = dantzig_enter_var(n, c_bar)
            
        # Compute descent direction
        u = B_inv@A[:,[enter_idx]]

        # Check nonnegativity of u
        if (u <= 0).all():
            print("\nNo component of u is positive. The optimal value is -infinity. No optimal solution attained.")
        else:
            # Choose an exiting variable
            if method == "Bland":
                # Apply Bland's rule
                argmin_idx, exit_idx = bland_exit_var(n, m, u, xB, B_idx)
            elif method == "Dantzig":
                # Apply Dantzig's rule
                argmin_idx, exit_idx = dantzig_exit_var(n, m, u, xB, B_idx)

        # Update B_idx - indices of the new basis
        B_idx[argmin_idx] = enter_idx

        # Update B_inv
        pivot = u[argmin_idx].item()
        other_idx = [k for k in range(n-m) if k!=argmin_idx]
        for k in other_idx:
            D = np.zeros((n-m,n-m))
            D[k,argmin_idx] = -u[k].item()/pivot
            Q = np.eye(n-m) + D
            B_inv = Q@B_inv
        B_inv[argmin_idx,:] /= pivot

        # Update values of the basic variables
        x, xB = basic_solution(n, B_inv, b, B_idx)
        path.append(x)

    return path

#%% Validation code
if __name__ == "__main__":
    seed = 1
    rng = np.random.Generator(np.random.MT19937(seed))
    #############################################################
    #### Algo validation (Bertsimas & Tsitsiklis, exple 3.5) ####
    #############################################################
    print("\nAlgorithm validation")
    # LP formulation (standard form)
    c = np.array([-10, -12, -12, 0, 0, 0]).reshape((-1,1))
    b = np.array([20, 20, 20]).reshape((-1,1))
    A = np.array([[1, 2, 2, 1, 0, 0],
                  [2, 1, 2, 0, 1, 0],
                  [2, 2, 1, 0, 0, 1]])

    # Other inputs
    method = "Dantzig"
    B_idx = [3, 4, 5] # indices of Aj columns to include in basis (Python indexing starts at 0)

    # Run revised Simplex algo
    path = revised_simplex(A, b, c, method, B_idx)
    display_path(path)

    #############################################################
    ################ Deterministic initial basis ################
    #############################################################
    print("\nDeterministic Initial Basis")
    # Inputs
    method = "Bland"
    N = 8
    A, b, c, B_idx = generate_problem(N, init_basis="slack")
    
    # Run revised Simplex algo
    path = revised_simplex(A, b, c, method, B_idx)
    display_path(path)

    ############################################################
    ################## Random initial basis ####################
    ############################################################
    print("\nRandom Initial Basis")
    # Inputs
    method = "Bland"
    N = 8
    
    # Run revised Simplex algo
    num_pts = 1000
    num_bases = []
    for _ in range(num_pts):
        A, b, c, B_idx = generate_problem(N, init_basis="random")
        path = revised_simplex(A, b, c, method, B_idx)
        num_bases.append(len(path))
    plt.hist(num_bases, bins="auto")
    plt.ylabel("Counts")
    plt.xlabel("Number of bases")
    plt.show()

    

    
