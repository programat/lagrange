import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt, sin, log, linspace, seterr, nanmin, \
    nanmax, inf, cos


matplotlib.use("TkAgg")

def lagrange(x, y, a): #x - value of argue y - value of funct a - point n - size of the table
    f = 0
    for i in range(len(y)):
        l = 1
        for j in range(len(x)):
            if i != j:
                l *= (a - x[j]) / (x[i] - x[j])
        l *= y[i]
        f += l
    return f

def select_function():
    print("\nList of functions:")
    print("1. y = sin(5x)", "2. 1 / (1 + 25 * (x^2))", "3. y = (exp(-x ** 2) + 2)",
          "4. y = sqrt(x ** 2 - 1)", "5. y = |x|", "6. y = x^2", sep="\n")
    number = input('Choose ur destiny (function): ')
    while number not in ['1', '2', '3', '4', '5', '6']:
        number = (input("Number incorrect, please try again: "))
    return int(number) - 1

def chebyshev_nodes(a, b, n):
    # n Chebyshev noder i intervallet [a, b]
    i = np.array(range(n))
    x = np.cos((2*i+1)*np.pi/(2*(n))) # noder over intervallet [-1,1]
    return 0.5*(b-a)*x+0.5*(b+a) # noder over intervallet [a,b]

func = [lambda x: sin(5*x),
        lambda x: 1 / (1 + 25 * (x ** 2)),
        lambda x: (exp(-x ** 2) + 2),
        lambda x: sqrt(x ** 2 - 1),
        lambda x: abs(x),
        lambda x: x**2]

# def Lagrange(x,y,t):
#     f=0
#     for j in range(len(y)):
#         p1=1; p2=1
#         for i in range(len(x)):
#             if i==j:
#                 p1=p1*1; p2=p2*1
#             else:
#                 p1=p1*(t-x[i])
#                 p2=p2*(x[j]-x[i])
#         f=f+y[j]*p1/p2
#     return f
if __name__ == '__main__':
    print('Give number of coordinate pairs (xi, yi)')
    n = int(input('n = '))
    number = select_function()

    print('What type of nodes you wanna use?\n1 - equally spaced points, 2 - Chebyshev nodes: ')
    que = input('1/2: ').replace(' ', '')
    while que not in ['1', '2']:
        number = input("Number incorrect, please try again: ")

    if que == '1':
        diff = (1 - (-1)) / (n - 1)
        nodes = [round(-1 + i * diff, 5) for i in range(n)]
        values = [func[number](nodes[i]) for i in range(len(nodes))]
    if que == '2':
        nodes = chebyshev_nodes(-1, 1, n)
        values = [func[number](nodes[i]) for i in range(len(nodes))]

    x = linspace(-1, 1)
    lagr = np.array([lagrange(nodes, values, i) for i in x], float)
    y = np.array([func[number](i) for i in x], float)




#something piece of shit
    # delta, n = float(input('delta = ')), int(input('n = '))
    # # delta, n = 0.1, 100
    #
    # # func = lambda x: sin(5*x)
    # # func = lambda x: 1 / (1 + 25 * (x ** 2))
    # # func = lambda x: (exp(-x ** 2) + 2)
    # func = lambda x: abs(x)
    # # func = lambda x: x**2
    #
    #
    # x = np.array([i*delta for i in range(int(-1/delta), int(1/delta)+1)], dtype='float')
    # # print(len(x))
    # y = np.array([func(x[i]) for i in range(len(x))], dtype='float')
    #
    # x_new = np.linspace(np.min(x), np.max(x), n)
    # # x_new = np.array(list(map(lambda x: np.cos((21 - 2 * x) / 22 * np.pi), range(n))))\
    # # x_new = [cos((step / steps * Math.PI) * (end - start) / 2)]
    #
    # y_new = [Lagrange(x, y, i) for i in x_new]


# output part
    print('Output on (1) or (2) graphs?')
    what = input('1/2: ').replace(' ', '')
    if what == '1':
        plt.plot(nodes, values, 'o', x, y, label='function')
        plt.suptitle('Lagrange and Function')
        plt.plot(nodes, values, 'o', x, lagr, color='green', markerfacecolor='red', label='lagrange')

        plt.grid(True)
        plt.legend()
        plt.show()

    if (what.replace(' ', '') == '2'):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(nodes, values, 'o', x, y, label='function')
        ax1.set_title('Lagrange')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(nodes, values, 'o', x, lagr, color='green', markerfacecolor='red', label='lagrange')
        ax2.grid(True)
        ax2.set_title('Function')
        fig.canvas.manager.set_window_title('Lagrange')
        ax2.legend()
        plt.show()