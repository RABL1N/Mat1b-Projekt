
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



def intersect_cell(i,j,a,N,rho,phi):

    x0 = a-((2*a*(i))/N)
    x1 = a-((2*a*(i))/N)+(2*a/N)
    y0 = a-((2*a*(j))/N)
    y1 = a-((2*a*(j))/N)+(2*a/N)

    x,y= symbols("x,y")
    eq = Eq(x*cos(phi)+y*sin(phi),rho)
    eq

    #return (x0,y0),(x1,y1)

    left = False
    right = False
    top = False
    bottom = False
    diagonal = False

    
    #left
    if (len(solve(eq,y))!=0) and (y0 < solve(eq,y)[0].subs({x:x0}) < y1):
        left = True

    #right
    if (len(solve(eq,y))!=0) and (y0 < solve(eq,y)[0].subs({x:x1}) < y1):
        right = True
        
    #top
    if (len(solve(eq,x))!=0) and (x0 < solve(eq,x)[0].subs({y:y1}) < x1):
        top = True
        
    #bottom
    if (len(solve(eq,x))!=0) and (x0 < solve(eq,x)[0].subs({y:y0}) < x1):
        bottom = True

    #diagonals
    if rho == 0 and (phi == 3*pi/4 or phi == 7*pi/4):
        diagonal = True
    
    return left, right, top, bottom, diagonal
    
    

def get_length(i, j, a, N, rho, phi):

    x,y= symbols("x,y")
    eq = Eq(x*cos(phi)+y*sin(phi),rho)

    x0 = a-((2*a*(i))/N)
    x1 = a-((2*a*(i))/N)+(2*a/N)
    y0 = a-((2*a*(j))/N)
    y1 = a-((2*a*(j))/N)+(2*a/N)

    if (len(solve(eq,y))!=0):
        # left intersection
        vy = solve(eq,y)[0].subs({x:x0})
        vx = x0
        
        # right intersection
        hy = solve(eq,y)[0].subs({x:x1})
        hx = x1

    if (len(solve(eq,x))!=0):
        # top intersection
        tx = solve(eq,x)[0].subs({y:y1})
        ty = y1

        # bottom intersection
        bx = solve(eq,x)[0].subs({y:y0})
        by = y0

    index = intersect_cell(i, j, a, N, rho, phi)

    if index[0] and index[1]:
        #print("left and right")
        vh = sqrt((vx-hx)**2+(vy-hy)**2)
        return vh

    if index[0] and index[2]:
        #print("left and top")
        vt = sqrt((vx-tx)**2+(vy-ty)**2)
        return vt
    
    if index[0] and index[3]:
        #print("left and bottom")
        vb = sqrt((vx-bx)**2+(vy-by)**2)
        return vb

    if index[1] and index[2]:
        #print("right and top")
        th = sqrt((tx-hx)**2+(ty-hy)**2)
        return th

    if index[1] and index[3]:
        #print("right and bottom")
        bh = sqrt((bx-hx)**2+(by-hy)**2)
        return bh

    if index[2] and index[3]:
        #print("top and bottom")
        tb = sqrt((tx-bx)**2+(ty-by)**2)
        return tb

    if index[4]:
        #print("diagonal")
        q0q1 = sqrt((x1-x0)**2+(y1-y0)**2)
        return q0q1
    

def construct_A(a, N, rho_list, phi_list):
    matrices = []

    for laser in range(len(rho_list)):
        rho = rho_list[laser]
        phi = phi_list[laser]
        cell_values = []  # Initialize cell_values inside the loop to reset it for each laser

        for i in range(N):  # row
            for j in range(N):  # col
                length = get_length(i, j, a, N, rho, phi)
                cell_values.append(length if length is not None else 0)

        M = Matrix(N, N, cell_values)  # Create a matrix from the current list of cell values
        matrices.append(M)  # Optionally store each matrix if you need to use all later
        print(M)  # Printing the matrix of the current laser

    return matrices  # Return the list of matrices if needed elsewhere





if __name__ == "__main__": # Run the code/functions from here

    init_printing(use_unicode=True)  # Aktiver pæn printning

    #print(intersect_cell(2,2,2,4,0,pi/4))

    #print(get_length(2,2,2,4,0.1,5*pi/3))

    construct_A(2, 4, [0.2,0.6,0.3,0.3], [4*pi/6,7*pi/4,8*pi/4,5*pi/4])



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