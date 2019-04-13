from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import cv2

class dummyEquations():
    def free(self):
        return ['z','x','y']

    def evaluate(self,l):
        return (3*l['x']**2*l['y']+3*l['x']*l['y']**2)
        #return ((3*((l['x']-50)*100)**2 - 3*((l['y']-50)*100) + 4)*np.sin((l['y']-50)*100))

class dummyEquations1():
    def free(self):
        return ['happy days','x']

    def evaluate(self,l):
        return (3*l['x']**2)

def graph(equation, final=False):
    freevars = equation.free()
    if len(freevars) == 2:
        xs = list(np.arange(100))
        ys = equation.evaluate({freevars[1]:np.arange(100)})
        fig = plt.figure()
        plt.xlabel(freevars[1])
        plt.ylabel(freevars[0])
        plt.plot(xs,ys)
        #plt.ylabel('some numbers')
        plt.show()
        if final:
            plt.show(block=False)
    if len(freevars) == 3:
        xs, ys = np.meshgrid(np.linspace(0,100,30),np.linspace(0,100,30))
        zs = equation.evaluate({freevars[1]:xs,freevars[2]:ys})
        fig = plt.figure()
        plt.subplot()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('surface')
        ax.set_xlabel(freevars[1])
        ax.set_ylabel(freevars[2])
        ax.set_zlabel(freevars[0])
        if final:
            plt.show(block=False)
    return fig

if __name__ == "__main__":
    p = graph(dummyEquations())
    p2 = graph(dummyEquations1(),True)
    input("Do Stuff")
    plt.close()