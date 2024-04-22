from sympy import *
init_printing()

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
    linsolve((A,b)), A, b

def opgave4():
    # Figur 4
    A = Matrix([[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1], # Vandret fra top til bund
                [1,0,0,1,0,0,1,0,0],[0,1,0,0,1,0,0,1,0],[0,0,1,0,0,1,0,0,1], # Lodret fra venstre til højre
                [0,1,0,1,0,0,0,0,0],[0,0,1,0,1,0,1,0,0],[0,0,0,0,0,1,0,1,0], # Top højre fra top til bund 
                [0,1,0,0,0,1,0,0,0],[1,0,0,0,1,0,0,0,1],[0,0,0,1,0,0,0,1,0]]) # Bund højre fra top til bund
    b = Matrix([5,3,5,2,3,8,2,3,4,3,4,3])
    linsolve((A,b)), A, b

def opgave7_1():
    # Opg 7
    mu, I, x = symbols("mu I x")
    mu = Function("mu")(x)
    I = Function("I")(x)

    diffeq = Eq(mu*I, diff(I))

    return dsolve(diffeq)
print(opgave7_1())

print("hello world")
#opgave 12
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f1 using numpy for handling vectorized operations over a grid
def f1(x, y):
    r = np.sqrt(x**2 + y**2)
    return np.where(r < 1, np.cos(np.pi / 2 * r), 0)

# Create a grid of x, y values
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x, y)

# Compute f1 on the grid
Z = f1(X, Y)

# Plot the surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('$f_1(x, y)$')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Plot of $f_1(x, y) = \\cos\\left(\\frac{\\pi}{2} \\sqrt{x^2 + y^2}\\right)$ if $x^2 + y^2 < 1$, else 0')

plt.show()
