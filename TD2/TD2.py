import numpy as np
import math

"""
def calculate_utility(V):
    Calculation of the utilities and the policies for each state
"""
def calculate_utility(V, V_prime, epsilon, gamma, R, T1, T2, T3, action, iterations):
    #Infinite loop in order to get to the solution
    while (1):
        #Loop on the states
        for i in range (len(V)):
                summ = np.matmul(T1[i][:], V_prime), np.matmul(T2[i][:], V_prime), np.matmul(T3[i][:], V_prime)
                max_summ = np.amax(summ)
                
                #Saves the value of pi
                action[i] = summ.index(max(summ))
                
                #Saves the latest value of V' to compare the error
                V[i] = V_prime[i]
                
                #Updates the Value of V'
                V_prime[i] = R[i] + gamma*max_summ
        iterations += 1
        #Stop criterion
        if (math.sqrt(np.sum((np.array(V) - np.array(V_prime))**2))/4 < epsilon):
            V = V_prime
            break
    return V, action, iterations

"""
def display(V, action, epsilon, iterations)
    Table to display the best policy: a verification has been done in the case
    of x = 0.25 (Q5) and also the case of x=0 (Q3)  
"""
def display(V, action, epsilon, iterations):
    print ("\nError value: %f \nNumber of iterations: %i\n"%(epsilon, iterations))
    print ("    |    S0   |    S1   |    S2   |    S3   |\n--------------------------------------------")
    print ("V*  | %2.4f | %2.4f | %2.4f | %2.4f | \n--------------------------------------------"%(V[0], V[1], V[2], V[3]))
    print ("pi* |    a%i   |    a%i   |    a%i   |    a%i   |"%(action[0], action[1], action[2], action[3]))

"""
main
"""
if __name__ == '__main__':
    #Definition of the parameters
    #Probability
    x = 0.25
    y = 0.25
    #Discounted factor
    gamma = 0.9
    #Maximum error expected
    epsilon = 1e-4
    #Transition Matrices
    T1 = [[0, 0, 0, 0],[0, 1-x, 0, x],[1-y, 0, 0, y],[1, 0, 0, 0]]
    T2 = [[0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
    T3 = [[0, 0, 1, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
    #Rewards
    R = [0, 0, 1, 10]
    V = [R[0], R[1], R[2], R[3]]
    V_prime = [R[0], R[1], R[2], R[3]]
    action = [0, 0, 0, 0]
    iterations = 0

    V, action, iterations = calculate_utility(V, V_prime, epsilon, gamma, R, T1, T2, T3, action, iterations)

    display(V, action, epsilon, iterations)