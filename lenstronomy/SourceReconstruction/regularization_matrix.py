import numpy as np

## here we define functions that give regularization matrix for source plane reconstruction

## below are three functions give zeroth-order, gradient and curvature regularizations for pixellated source plane reconstruction respectively
    # input1, number of coloums, i.e. source region width of the x-direction
    # input2, number of rows, i.e. source region width of the y-direction
    # output is a col * row matrix
    
def give_zeroth_order_regularization_pixel(col,row):
    reg = np.identity(col*row)
    return reg

def give_gradient_regularization_pixel(col,row):
    if col > 5 and row >5:
        reg = give_gradient_regularization_rect_large(col,row)
    else:
        reg = give_gradient_regularization_rect(col,row)
    return reg

def give_curvature_regularization_pixel(col,row):
    if col > 5 and row >5:
        reg = give_curvature_regularization_rect_large(col,row)
    else:
        reg = give_curvature_regularization_rect(col,row)
    return reg

## below are functions give the gradient and curvature regularization matrix for the shapelets of lenstronomy
    # input is the n_max defined for a set of shaplets
    # output is a num_of_shapelets-by-num_of_shapelets size matrix, where the num_of_shapelets is calculated from n_max
def give_curvature_regularization_shapelets(n_max):
    num_of_shapelets = int( ((n_max+1)*(n_max+2))/2 )
    Cv = np.zeros(((num_of_shapelets,num_of_shapelets)))

    Cv_dict = give_curvature_regularization_dictionary_for_SHO_wavefunction(n_max)

    seq1 = 0
    for i in range(n_max+1):
        for j in range(i+1):
            n_temp = j
            m_temp = i-j
            seq2 = 0

            for i1 in range(n_max+1):
                for j1 in range(i1+1):
                    k_temp = j1
                    l_temp = i1 - j1

                    Cv[seq1,seq2] = Cv_dict[n_temp,m_temp,k_temp,l_temp]
                    seq2 += 1

            seq1 += 1
    return Cv

def give_gradient_regularization_shapelets(n_max):
    num_of_shapelets = int( ((n_max+1)*(n_max+2))/2 )
    Cv = np.zeros(((num_of_shapelets,num_of_shapelets)))

    Cv_dict = give_gradient_regularization_dictionary_for_SHO_wavefunction(n_max)

    seq1 = 0
    for i in range(n_max+1):
        for j in range(i+1):
            n_temp = j
            m_temp = i-j
            seq2 = 0

            for i1 in range(n_max+1):
                for j1 in range(i1+1):
                    k_temp = j1
                    l_temp = i1 - j1

                    Cv[seq1,seq2] = Cv_dict[n_temp,m_temp,k_temp,l_temp]
                    seq2 += 1

            seq1 += 1
    return Cv


## Auxiliary functions

def give_gradient_regularization_rect(n,m):  # give gradient regularization matrix, slow when n and m are large
    Lambda = np.zeros((n*m,n*m))
    s = np.zeros((n+1,m+1,n+1,m+1))

    s[1,1,1,1] += 2
    s[1,m,1,m] += 2
    s[n,1,n,1] += 2
    s[n,m,n,m] += 2

    for i0 in range(n-2):
        i = i0 + 2
        s[i,1,i,1] += 1
        s[i,m,i,m] += 1
        
    for i0 in range(m-2):
        i = i0 + 2
        s[1,i,1,i] += 1
        s[n,i,n,i] += 1
                
    for i0 in range(n-1):
        i = i0 + 1
        for j0 in range(m):
            j = j0 + 1
            s[i,j,i,j] += 1
            s[i+1,j,i+1,j] += 1
            s[i,j,i+1,j] += -1
            s[i+1,j,i,j] += -1
    
    for i0 in range(m-1):
        i = i0 + 1
        for j0 in range(n):
            j = j0 + 1
            s[j,i,j,i] += 1
            s[j,i+1,j,i+1] += 1
            s[j,i,j,i+1] += -1
            s[j,i+1,j,i] += -1

    for i0 in range(n):
        i = i0 +1
        for j0 in range(m):
            j = j0 +1
            for k0 in range(n):
                k = k0 +1
                for l0 in range(m):
                    l = l0 +1

                    Lambda_x = i + n*(j-1) -1
                    Lambda_y = k + n*(l-1) -1

                    Lambda[Lambda_x,Lambda_y] = s[i,j,k,l]
                    
    return Lambda

def give_gradient_regularization_rect_large(n,m):  # for gradient regularization, faster for large n and m
    Lambda = np.zeros((n*m,n*m))
    
    block0 = 4 * np.identity(n)
    for i in range(n-1):
        block0[i,i+1] = -1
        block0[i+1,i] = -1
    
    block1 = -1 * np.identity(n)
    
    for i in range(m):
        Lambda[n*i:n*i+n:1,n*i:n*i+n:1] = block0
    for i in range(m-1):
        Lambda[n*i+n:n*i+2*n:1,n*i:n*i+n:1] = block1
        Lambda[n*i:n*i+n:1,n*i+n:n*i+2*n:1] = block1
    
    return Lambda

def give_curvature_regularization_rect(n,m):  
    # n,m are the source region col and row numbers. It is slow when n and m are large
    Lambda = np.zeros((n*m,n*m))
    s = np.zeros((n+1,m+1,n+1,m+1))

    for i in range(n):
        s[i+1,1,i+1,1] += 1
        s[i+1,m,i+1,m] += 1
    for i in range(m):
        s[1,i+1,1,i+1] += 1
        s[n,i+1,n,i+1] += 1
    
    s[2,1,2,1] += 1
    s[1,1,1,1] += 4
    s[1,1,2,1] += -2
    s[2,1,1,1] += -2
    s[1,2,1,2] += 1
    s[1,1,1,1] += 4
    s[1,2,1,1] += -2
    s[1,1,1,2] += -2

    s[n-1,1,n-1,1] += 1
    s[n,1,n,1] += 4
    s[n-1,1,n,1] += -2
    s[n,1,n-1,1] += -2
    s[n,2,n,2] += 1
    s[n,1,n,1] += 4
    s[n,2,n,1] += -2
    s[n,1,n,2] += -2

    s[2,m,2,m] += 1
    s[1,m,1,m] += 4
    s[2,m,1,m] += -2
    s[1,m,2,m] += -2
    s[1,m-1,1,m-1] += 1
    s[1,m,1,m] += 4
    s[1,m-1,1,m] += -2
    s[1,m,1,m-1] += -2

    s[n-1,m,n-1,m] += 1
    s[n,m,n,m] += 4
    s[n-1,m,n,m] += -2
    s[n,m,n-1,m] += -2
    s[n,m-1,n,m-1] += 1
    s[n,m,n,m] += 4
    s[n,m-1,n,m] += -2
    s[n,m,n,m-1] += -2

    for i0 in range(n-2):
        i = i0 + 2

        s[i-1,1,i-1,1] += 1
        s[i+1,1,i+1,1] += 1
        s[i,1,i,1] += 4
        s[i-1,1,i+1,1] += 1
        s[i+1,1,i-1,1] += 1
        s[i-1,1,i,1] += -2
        s[i,1,i-1,1] += -2
        s[i+1,1,i,1] += -2
        s[i,1,i+1,1] += -2

        s[i,2,i,2] += 1
        s[i,1,i,1] += 4
        s[i,2,i,1] += -2
        s[i,1,i,2] += -2

        s[i-1,m,i-1,m] += 1
        s[i+1,m,i+1,m] += 1
        s[i,m,i,m] += 4
        s[i-1,m,i+1,m] += 1
        s[i+1,m,i-1,m] += 1
        s[i-1,m,i,m] += -2
        s[i,m,i-1,m] += -2
        s[i+1,m,i,m] += -2
        s[i,m,i+1,m] += -2

        s[i,m-1,i,m-1] += 1
        s[i,m,i,m] += 4
        s[i,m-1,i,m] += -2
        s[i,m,i,m-1] += -2
        
        for j0 in range(m-2):
            j = j0 + 2

            s[i,j-1,i,j-1] += 1
            s[i,j+1,i,j+1] += 1
            s[i,j,i,j] += 4
            s[i,j-1,i,j+1] += 1
            s[i,j+1,i,j-1] += 1
            s[i,j-1,i,j] += -2
            s[i,j,i,j-1] += -2
            s[i,j+1,i,j] += -2
            s[i,j,i,j+1] += -2

            s[i-1,j,i-1,j] += 1
            s[i+1,j,i+1,j] += 1
            s[i,j,i,j] += 4
            s[i-1,j,i+1,j] += 1
            s[i+1,j,i-1,j] += 1
            s[i-1,j,i,j] += -2
            s[i,j,i-1,j] += -2
            s[i+1,j,i,j] += -2
            s[i,j,i+1,j] += -2
        
    for i0 in range(m-2):
        i = i0 + 2

        s[1,i-1,1,i-1] += 1
        s[1,i+1,1,i+1] += 1
        s[1,i,1,i] += 4
        s[1,i-1,1,i+1] += 1
        s[1,i+1,1,i-1] += 1
        s[1,i-1,1,i] += -2
        s[1,i,1,i-1] += -2
        s[1,i+1,1,i] += -2
        s[1,i,1,i+1] += -2

        s[2,i,2,i] += 1
        s[1,i,1,i] += 4
        s[2,i,1,i] += -2
        s[1,i,2,i] += -2

        s[n,i-1,n,i-1] += 1
        s[n,i+1,n,i+1] += 1
        s[n,i,n,i] += 4
        s[n,i-1,n,i+1] += 1
        s[n,i+1,n,i-1] += 1
        s[n,i-1,n,i] += -2
        s[n,i,n,i-1] += -2
        s[n,i+1,n,i] += -2
        s[n,i,n,i+1] += -2

        s[n-1,i,n-1,i] += 1
        s[n,i,n,i] += 4
        s[n-1,i,n,i] += -2
        s[n,i,n-1,i] += -2

    for i0 in range(n):
        i = i0 +1
        for j0 in range(m):
            j = j0 +1
            for k0 in range(n):
                k = k0 +1
                for l0 in range(m):
                    l = l0 +1

                    Lambda_x = i + n*(j-1) -1
                    Lambda_y = k + n*(l-1) -1

                    Lambda[Lambda_x,Lambda_y] = s[i,j,k,l]
                    
    return Lambda

def give_curvature_regularization_rect_large(n,m):  
    # gives the same result as the give_curvature_regularization_rect(n,m) function when n and m are large, and faster
    Lambda = np.zeros((n*m,n*m))
    
    block0 = np.zeros((n,n))
    for i in range(n):
        block0[i,i] = 12
    for i in range(n-1):
        block0[i,i+1] = -4
        block0[i+1,i] = -4
    for i in range(n-2):
        block0[i,i+2] = 1
        block0[i+2,i] = 1
        
    block1 = -4*np.identity(n)
    block2 = np.identity(n)
    
    for i in range(m):
        Lambda[n*i:n*i+n:1,n*i:n*i+n:1] = block0
    for i in range(m-1):
        Lambda[n*i+n:n*i+2*n:1,n*i:n*i+n:1] = block1
        Lambda[n*i:n*i+n:1,n*i+n:n*i+2*n:1] = block1
    for i in range(m-2):
        Lambda[n*i+2*n:n*i+3*n:1,n*i:n*i+n:1] = block2
        Lambda[n*i:n*i+n:1,n*i+2*n:n*i+3*n:1] = block2
    
    return Lambda

def give_curvature_regularization_dictionary_for_SHO_wavefunction(n_max):
    # Auxiliary function for shapelets curvature regularization
    # input is n_max, which is the largest n of the defined set of shapelets of lenstronomy
    # output is a 4-dimensional matrix, gives the coefficient of \phi_{n,m}\phi_{k,l} term contributing to the regularization term
    S = np.zeros((n_max+1,n_max+1,n_max+1,n_max+1))
    
    for n in range(n_max+1):
        for m in range(n_max+1):
            
            S[n,m,n,m] += (8*m*n + 6*n*n + 6*m*m + 10*m + 10*n + 8)
            
            if (n-2) >= 0:
                S[n,m,n-2,m] += (-4 * np.sqrt(n*(n-1))*(m+n))
                
                if (n-4) >= 0:
                    S[n,m,n-4,m] += np.sqrt(n*(n-1)*(n-2)*(n-3))
                if (m-2) >= 0:
                    S[n,m,n-2,m-2] += 2*np.sqrt(n*m*(n-1)*(m-1))
                if (m+2) <= n_max:
                    S[n,m,n-2,m+2] += 2*np.sqrt(n*(n-1)*(m+1)*(m+2))
                    
            if (n+2) <= n_max:
                S[n,m,n+2,m] += (-4*np.sqrt((n+1)*(n+2))*(n+m+2))
                
                if (n+4) <= n_max:
                    S[n,m,n+4,m] += np.sqrt((n+1)*(n+2)*(n+3)*(n+4))
                if (m-2) >= 0:
                    S[n,m,n+2,m-2] += 2*np.sqrt((n+1)*(n+2)*m*(m-1))
                if (m+2) <= n_max:
                    S[n,m,n+2,m+2] += 2*np.sqrt((m+1)*(m+2)*(n+1)*(n+2))
                    
            if (m-2) >=0:
                S[n,m,n,m-2] += -4 * np.sqrt(m*(m-1))*(n+m)
                
                if (m-4) >=0:
                    S[n,m,n,m-4] += np.sqrt(m*(m-1)*(m-2)*(m-3))
            
            if (m+2) <= n_max:
                S[n,m,n,m+2] += -4 * np.sqrt((m+1)*(m+2))*(n+m+2)
                
                if (m+4) <= n_max:
                    S[n,m,n,m+4] += np.sqrt((m+1)*(m+2)*(m+3)*(m+4))
                    
    S = (1/4) * S
                           
    return S

def give_gradient_regularization_dictionary_for_SHO_wavefunction(n_max):
    # Auxiliary function for shapelets gradient regularization
    # input is n_max, which is the largest n of the defined set of shapelets of lenstronomy
    # output is a 4-dimensional matrix, gives the coefficient of \phi_{n,m}\phi_{k,l} term contributing to the regularization term
    S = np.zeros((n_max+1,n_max+1,n_max+1,n_max+1))
    
    for n in range(n_max+1):
        for m in range(n_max+1):
            
            S[n,m,n,m] += (2*n+2*m+2)
            
            if (n-2) >= 0:
                S[n,m,n-2,m] += -np.sqrt(n*(n-1))

            if (n+2) <= n_max:
                S[n,m,n+2,m] += -np.sqrt((n+1)*(n+2))
                                   
            if (m-2) >=0:
                S[n,m,n,m-2] += -np.sqrt(m*(m-1))
                            
            if (m+2) <= n_max:
                S[n,m,n,m+2] += -np.sqrt((m+1)*(m+2))
                
    S = (1/2) * S
                           
    return S