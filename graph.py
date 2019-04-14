from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import cv2
#plt.ion()
plt.show()

class dummyEquations():
    def get_variables(self):
        return ['z','x','y']

    def evaluate(self,l):
        return (3*l['x']**2*l['y']+3*l['x']*l['y']**2)
        #return ((3*((l['x']-50)*100)**2 - 3*((l['y']-50)*100) + 4)*np.sin((l['y']-50)*100))

class dummyEquations1():
    def get_variables(self):
        return ['happy days','x']

    def evaluate(self,l):
        return (3*l['x']**2)

def graph(equation, final=False):

    if equation:
        freevars = equation.get_variables()
        plt.clf()
        plt.cla()
        plt.ion()
        if len(freevars) == 2:
            xs = list(np.arange(100))
            ys = equation.evaluate([np.arange(100)])
            #fig = plt.figure()
            plt.xlabel(freevars[1])
            plt.ylabel(freevars[0])
            plt.plot(xs,ys)
            #plt.ylabel('some numbers')
            #plt.show()
            plt.draw()
            plt.pause(0.001)
        if len(freevars) == 3:
            xs, ys = np.meshgrid(np.linspace(0,100,30),np.linspace(0,100,30))
            zs = equation.evaluate([xs,ys])
            plt.subplot()
            ax = plt.axes(projection='3d')
            ax.plot_surface(xs, ys, zs, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            ax.set_title('surface')
            ax.set_xlabel(freevars[1])
            ax.set_ylabel(freevars[2])
            ax.set_zlabel(freevars[0])
            plt.draw()
            plt.pause(0.001)
        plt.ioff()
        return plt

if __name__ == "__main__":
    p = graph(dummyEquations())
    p2 = graph(dummyEquations1(),True)
    input("Do Stuff")
    plt.close()