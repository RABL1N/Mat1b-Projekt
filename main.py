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