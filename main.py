
from sympy import *
init_printing()

#init_printing(use_unicode=True)  # Gør output pænere med unicode-symboler


#Opgave 12
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def opgave2():
    # Opg 2 og 3
    A = Matrix([[1, 1, 0, 0],[0, 0, 1, 1],[1, 0, 1, 0],[0, 1, 0, 1]])
    b = Matrix([5, 2, 3, 4])
    A,b
    return(linsolve((A,b)))

def opgave3():
    # Figur 3
    A = Matrix([[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1],[1,0,0,1,0,0,1,0,0],[0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1]])
    b = Matrix([3,5,1,3,3,3])
    return linsolve((A,b)), A, b

def opgave4():
    # Figur 4
    A = Matrix([[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1], # Vandret fra top til bund
                [1,0,0,1,0,0,1,0,0],[0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1], # Lodret fra venstre til højre
                [0,1,0,1,0,0,0,0,0],[0,0,1,0,1,0,1,0,0],[0,0,0,0,0,1,0,1,0], # Top højre fra top til bund 
                [0,1,0,0,0,1,0,0,0],[1,0,0,0,1,0,0,0,1],[0,0,0,1,0,0,0,1,0]]) # Bund højre fra top til bund
    b = Matrix([5,3,5,2,3,8,2,3,4,3,4,3])
    return linsolve((A,b)), A, b

def opgave7_1():
    # Opg 7
    mu, I, x = symbols("mu I x")
    mu = Function("mu")(x)
    I = Function("I")(x)

    diffeq = Eq(mu*I, diff(I))

    return dsolve(diffeq)


def opgave10():
    x,y,t,rho,theta = symbols("x,y,t,tho,theta")

    #r_t = rho*Matrix([cos(theta), sin(theta)])+t*Matrix([-sin(theta),cos(theta)])
    r_t = Matrix(rho*cos(theta)+t*(-sin(theta)),rho*sin(theta)+t*cos(theta))

    print(r_t)


def opgave11():
    # Symbolic expressions for a and b
    a = -sqrt(3)/3
    b = simplify(sqrt(3)/9)  # Assuming this is the exact form of b

    # Coordinates of the point you're interested in
    x1 = 0
    y1 = 0

    # Calculating rho symbolically
    rho = abs(a*x1+b-y1)/sqrt(a**2+1)

    # Simplifying rho
    simplified_rho = simplify(rho)

    # Display the result
    print(simplified_rho)



def f1(x, y):
    # Define the function f1 using numpy for handling vectorized operations over a grid
    r = np.sqrt(x**2 + y**2)
    return np.where(r < 1, np.cos(np.pi / 2 * r), 0)


def opgave29_1():
    sol = opgave4()[1]

    vertical = Matrix([sol.row(i) for i in range(0, 3)]) 

    horizontal = Matrix([sol.row(i) for i in range(3, 6)]) 

    left_diagonals = Matrix([sol.row(i) for i in range(6, 9)]) 

    right_diagonals = Matrix([sol.row(i) for i in range(9, 12)]) 


    A = Matrix([vertical,horizontal,left_diagonals*sqrt(2),right_diagonals*sqrt(2)])
    b = Matrix([5,3,5,2,3,8,2*sqrt(2),3*sqrt(2),4*sqrt(2),3*sqrt(2),4*sqrt(2),3*sqrt(2)])
    return linsolve((A,b)), A, b

def opgave29_2():
    sol = opgave4()[1]

    vertical = Matrix([sol.row(i) for i in range(0, 3)]) 

    horizontal = Matrix([sol.row(i) for i in range(3, 6)]) 

    left_diagonals = Matrix([sol.row(i) for i in range(6, 9)]) 

    right_diagonals = Matrix([sol.row(i) for i in range(9, 12)]) 


    A = Matrix([vertical,horizontal,left_diagonals*sqrt(2),right_diagonals*sqrt(2)])
    b = Matrix([5,3,5,2,3,8,2,3,4,3,4,3])
    return linsolve((A,b)), A, b

def opgave29_3():
    sol = opgave4()[1]

    vertical = Matrix([sol.row(i) for i in range(0, 3)]) 

    horizontal = Matrix([sol.row(i) for i in range(3, 6)]) 

    left_diagonals = Matrix([sol.row(i) for i in range(6, 9)]) 

    right_diagonals = Matrix([sol.row(i) for i in range(9, 12)]) 


    A = Matrix([vertical,horizontal,left_diagonals,right_diagonals])
    b = Matrix([5,3,5,2,3,8,2*sqrt(2),3*sqrt(2),4*sqrt(2),3*sqrt(2),4*sqrt(2),3*sqrt(2)])
    return linsolve((A,b)), A, b


def opgave29_4():
    sol = opgave4()[1]

    vertical = Matrix([sol.row(i) for i in range(0, 3)]) 

    horizontal = Matrix([sol.row(i) for i in range(3, 6)]) 

    left_diagonals = Matrix([sol.row(i) for i in range(6, 9)]) 

    right_diagonals = Matrix([sol.row(i) for i in range(9, 12)]) 


    A = Matrix([vertical,horizontal,left_diagonals,right_diagonals])
    b = Matrix([5,3,5,2,3,8,2,3,4,3,4,3])
    return linsolve((A,b)), A, b



if __name__ == "__main__": # Run the code/functions from here

    init_printing(use_unicode=True)  # Aktiver pæn printning

    #---------------- Opgave 29 ----------------
    
    #pretty_print(opgave29())
    #pretty_print(opgave29_1())

    

    #---------------- Opgave 11 ----------------
    #opgave11()



    #---------------- Opgave 12 ----------------

    # # Create a grid of x, y values
    # x = np.linspace(-1.5, 1.5, 400)
    # y = np.linspace(-1.5, 1.5, 400)
    # X, Y = np.meshgrid(x, y)

    # # Compute f1 on the grid
    # Z = f1(X, Y)

    # # Plot the surface
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # cbar.set_label('$f_1(x, y)$')

    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.set_title('Plot of $f_1(x, y) = \\cos\\left(\\frac{\\pi}{2} \\sqrt{x^2 + y^2}\\right)$ if $x^2 + y^2 < 1$, else 0')

    # plt.show()