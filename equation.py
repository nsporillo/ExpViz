from abc import ABC, abstractmethod
import numpy as np


class Function(ABC):
    def __init__(self, components):
        self.components = components
        self.variables = set()

    def __repr__(self):
        return repr(self.variables)

    def get_variables(self):
        for component in self.components:
            self.variables.update(component.get_variables())
        self.update_variable_indxs()
        return self.variables

    def update_variable_indxs(self, vars=[]):
        if vars:
            self.variables = vars
        else:
            self.variables = sorted(list(self.variables))
        for component in self.components:
            component.update_variable_indxs(self.variables)

    @abstractmethod
    def evaluate(self, vals):
        yield None


class Sum(Function):
    def __init__(self, f1, f2):
        super().__init__([f1, f2])

    def __repr__(self):
        return "(" + repr(self.components[0]) + " + " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        sum = 0
        for component in self.components:
            sum += component.evaluate(vals)
        return sum


class Difference(Function):
    def __init__(self, f1, f2):
        super().__init__([f1, f2])

    def __repr__(self):
        return "(" + repr(self.components[0]) + " - " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return self.components[0].evaluate(vals) - self.components[1].evaluate(vals)


class Product(Function):
    def __init__(self, f1, f2):
        super().__init__([f1, f2])

    def __repr__(self):
        return "(" + repr(self.components[0]) + " * " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        prod = self.components[0].evaluate(vals)
        for component in self.components[1:]:
            prod *= component.evaluate(vals)
        return prod


class Quotient(Function):
    def __init__(self, f1, f2):
        super().__init__([f1, f2])

    def __repr__(self):
        return "(" + repr(self.components[0]) + " / " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return self.components[0].evaluate(vals) / self.components[1].evaluate(vals)


class Modulus(Function):
    def __init__(self, f1, f2):
        super().__init__([f1, f2])

    def __repr__(self):
        return "(" + repr(self.components[0]) + " % " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return np.mod(self.components[0].evaluate(vals), self.components[1].evaluate(vals))


class Sine(Function):
    def __init__(self, f1):
        super().__init__([f1])

    def __repr__(self):
        return "(Sin(" + repr(self.components[0]) + "))"

    def evaluate(self, vals):
        return np.sin(self.components[0].evaluate(vals))


class Cosine(Function):
    def __init__(self, f1):
        super().__init__([f1])

    def __repr__(self):
        return "(Cos(" + repr(self.components[0]) + "))"

    def evaluate(self, vals):
        return np.cos(self.components[0].evaluate(vals))


class Exponent(Function):
    def __init__(self, f1, f2):
        super().__init__([f1, f2])

    def __repr__(self):
        return "(" + repr(self.components[0]) + " ^ " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return np.power(self.components[0].evaluate(vals), self.components[1].evaluate(vals))


class Logarithm(Function):
    def __init__(self, f1):
        super().__init__([f1])

    def __repr__(self):
        return "(ln(" + repr(self.components[0]) + "))"

    def evaluate(self, vals):
        return np.log(self.components[0].evaluate(vals))


class Constant(Function):
    def __init__(self, c):
        super().__init__([c])

    def __repr__(self):
        return "(" + repr(self.components[0]) + ")"

    def get_variables(self):
        return set()

    def update_variable_indxs(self, vars=[]):
        if vars:
            self.variables = vars
        else:
            self.variables = sorted(list(self.variables))

    def evaluate(self, vals):
        return self.components[0]


class Variable(Function):
    def __init__(self, name):
        super().__init__([name])
        self.variables.update(self.components)
        self.var_list_idx = 0

    def __repr__(self):
        return "(" + repr(self.components[0]) + ")"

    def get_variables(self):
        return self.variables

    def update_variable_indxs(self, vars=[]):
        if vars:
            self.variables = vars
        else:
            self.variables = sorted(list(self.variables))
        self.var_list_idx = self.variables.index(self.components[0])

    def evaluate(self, vals):
        return vals[self.var_list_idx]
