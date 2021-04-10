import numpy as np
import math

def generate_all_mu(betas, box):
    M = [] # 0 - mu_u | 1 - mu_l
    for beta in betas:
        M.append([beta - box, beta + box])
    return np.array(M)

def median_of_medians(elems):     
    sublists = [elems[j:j+5] for j in range(0, len(elems), 5)]     
    medians = []     
    for sublist in sublists:         
        medians.append(sorted(sublist)[len(sublist)//2])     
    if len(medians) <= 5:         
        return sorted(medians)[len(medians)//2]     
    else:         
        return median_of_medians(medians) 

def generate_betas(mu, betas, box, M):
    new_betas = []
    for i in range(len(betas)):
        if mu < M[i][0]: # C
            new_betas.append(box)   
        elif mu > M[i][1]: # -C
            new_betas.append(-box) 
        else: # beta_i - mu
            new_betas.append(float(betas[i] - mu)) 
    return np.array(new_betas)

def lin_interp(mu_L, mu_U, betas, box, M):
    h_L = np.sum(generate_betas(mu_L, betas, box, M))
    h_U = np.sum(generate_betas(mu_U, betas, box, M))
    return mu_L - h_L*((mu_U-mu_L)/(h_U-h_L))

def adjust_M(M, mu, mode):
    # mode 0 | retain only all mu_l and mu_u bigger than mu (sum is too high, we need to lower it)
    # mode 1 | retain only all mu_l and mu_u lower than mu (sum is too low, we need to higher it)
    return M[M > mu] if mode == 0 else M[M < mu]

def solveKP(box, linear_constraint, betas, verbose = False):
    """
    box               : range constraining the betas (-box box) = (-C C)
    linear_constraint : value which must be sum of betas
    betas             : input variables
    """
    betas = np.hstack(betas)
    original_M = generate_all_mu(betas, box) # structured list - each element is a pair (0 for mu_u and 1 for mu_l)
    mu_L, mu_U = math.inf, -math.inf
    M = np.ravel(original_M) # copy and unroll original M
    if verbose:
        print(f"INITIAL BETAS: {betas}\n INITIAL mu_L: {mu_L}\nINITIAL mu_U: {mu_U}")
    while M.size != 0:
        mu = median_of_medians(M)
        temp_betas = generate_betas(mu, betas, box, original_M)
        betas_sum = np.sum(temp_betas)
        
        if verbose:
            print(f"\nMEDIAN OF {M} IS {mu}")
            print(f"NEW BETAS: {temp_betas}")
            print(f"SUM OF BETAS: {betas_sum}")
        
        if betas_sum == linear_constraint: # LUCKY ESCAPE!
            return np.vstack(temp_betas)
        elif betas_sum > linear_constraint:
            mu_L = mu
            M = adjust_M(M, mu, 0)
        else:
            mu_U = mu
            M = adjust_M(M, mu, 1)
        
        if verbose:
            print(f"END OF ITER mu_L {mu_L} AND mu_U {mu_U}")

    # if we arrive here then solution stands in linear interpolation
    if verbose:
        print("SOLUTION FOUND BY LINEAR INTERPOLATION")
    mu = lin_interp(mu_L, mu_U, betas, box, original_M)
    return np.vstack(generate_betas(mu, betas, box, original_M))

if __name__ == "__main__":
    C = 3
    box = C
    linear_constraint = 0
    betas = np.random.uniform(-1,1,1337)
    betas = solveKP(box, linear_constraint, betas)
    print(f"FINAL BETAS: {betas}\nFINAL SUM OF BETAS: {np.sum(betas)}")
