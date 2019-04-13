from abc import ABC, abstractmethod
import math


class Function(ABC):
    def __init__(self):
        self.components = []
        self.variables = []

    @abstractmethod
    def evaluate(self, vals):
        yield None


class Sum(Function):
    def __init__(self, f1, f2):
        super().__init__()
        self.components = [f1, f2]

    def __repr__(self):
        return "(" + repr(self.components[0]) + " + " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        sum = 0
        for component in self.components:
            sum += component.evaluate(vals)
        return sum


class Difference(Function):
    def __init__(self, f1, f2):
        super().__init__()
        self.components = [f1, f2]

    def __repr__(self):
        return "(" + repr(self.components[0]) + " - " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return self.components[0].evaluate(vals) - self.components[1].evaluate(vals)


class Product(Function):
    def __init__(self, f1, f2):
        super().__init__()
        self.components = [f1, f2]

    def __repr__(self):
        return "(" + repr(self.components[0]) + " * " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        prod = 1
        for component in self.components:
            prod *= component.evaluate(vals)
        return prod


class Quotient(Function):
    def __init__(self, f1, f2):
        super().__init__()
        self.components = [f1, f2]

    def __repr__(self):
        return "(" + repr(self.components[0]) + " / " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return self.components[0].evaluate(vals) / self.components[1].evaluate(vals)


class Modulus(Function):
    def __init__(self, f1, f2):
        super().__init__()
        self.components = [f1, f2]

    def __repr__(self):
        return "(" + repr(self.components[0]) + " % " + repr(self.components[1]) + ")"

    def evaluate(self, vals):
        return int(self.components[0].evaluate(vals)) % int(self.components[1].evaluate(vals))


class Sine(Function):
    def __init__(self, f1):
        super().__init__()
        self.components = [f1]

    def evaluate(self, vals):
        return math.sin(self.components[0].evaluate(vals))


class Cosine(Function):
    def __init__(self, f1):
        super().__init__()
        self.components = [f1]

    def evaluate(self, vals):
        return math.cos(self.components[0].evaluate(vals))


class Exponent(Function):
    def __init__(self, f1, f2):
        super().__init__()
        self.components = [f1, f2]

    def evaluate(self, vals):
        return math.pow(self.components[0].evaluate(vals), self.components[1].evaluate(vals))


class Logarithm(Function):
    def evaluate(self, vals):
        return math.log(self.components[0].evaluate(vals))


class Constant(Function):
    def __init__(self, c):
        super().__init__()
        self.components = [c]

    def __repr__(self):
        return "(" + repr(self.components[0]) + ")"

    def evaluate(self, vals):
        return self.components[0]


class Variable(Function):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.components = [name]

    def __repr__(self):
        return "(" + repr(self.components[0]) + ")"

    def evaluate(self, vals):
        return vals[self.components[0]]

    def get_name(self):
        return self.name
