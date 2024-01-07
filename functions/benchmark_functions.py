import numpy as np
import math

def ackley(a, b, c, d, x):
    """
    Following: https://www.sfu.ca/~ssurjano/ackley.html
    """

    sum1 = 0
    sum2 = 0

    for i in range(d):
        sum1 += x[i] ** 2
        sum2 += np.cos(c * x[i])
    
    value = -a * np.exp(-b * math.sqrt((1/d) * sum1)) - np.exp((1/d) * sum2) + a + np.exp(1)

    return value

def rastrigin(d, x):
    """
    Following: https://www.sfu.ca/~ssurjano/rastr.html
    """

    sum = 0
    for i in range(d):
        sum += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
    
    value = 10*d + sum
    return value

def schwefel(d, x):
    """
    Following: https://www.sfu.ca/~ssurjano/schwef.html
    """

    sum = 0
    for i in range(d):
        sum += x[i] * np.sin(math.sqrt(x[i]))
    
    value = 418.9829 * d - sum

    return value

def rosenbrock(d, x):
    """
    Following: https://www.sfu.ca/~ssurjano/rosen.html
    """

    sum = 0
    for i in range(d - 1):
        sum += ( 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2 )
    
    return sum
