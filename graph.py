from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

class dummyEquations():
    def free(self):
        return 2

    def evaluate(self,l):
        return (3*l[0]**2 - 3*l[1] + 4)*np.sin(l[1])

def graph(equation):
    freevars = equation.free()
    if freevars == 1:
        xs = list(np.arange(100))
        ys = sorted(list(map(equation.evaluate,(np.arange(100).reshape((100,1)).tolist()))))
        plt.plot(xs,ys)
        plt.ylabel('some numbers')
        plt.show()
    if freevars == 2:
        xs, ys = np.meshgrid(np.arange(100),np.arange(100))
        zs = equation.evaluate([xs,ys])
        ax = plt.axes(projection='3d')
        ax.contour3D(xs, ys, zs, 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

graph(dummyEquations())