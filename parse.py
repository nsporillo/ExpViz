import sys
import re
import equation as eq
import numpy as np

class Unit:
    def __init__(self, left, term):
        self.left = left
        self.term = term

    def __repr__(self):
        return str(self.left) + " -> " + self.term


class Production:
    def __init__(self, left, A, B):
        self.left = left
        self.A = A
        self.B = B

    def __repr__(self):
        return str(self.left) + " -> " + str(self.A) + " " + str(self.B)


class Rule:
    def __init__(self, left, terms):
        self.left = left
        self.terms = terms

    def __repr__(self):
        return repr(self.left) + " -> " + repr(self.terms)

    def __eq__(self, other):
        return repr(self).eq(repr(other))

    def __hash__(self):
        return hash(repr(self))


class Grammar:
    def __init__(self):
        self.rules = []
        self.terms = []
        self.nterms = dict()
        self.start = 0

    def __len__(self):
        return len(self.nterms)

    def __repr__(self):
        res = "TERMINALS:\n"
        for term in self.terms:
            res += repr(term) + "\n"
        res += "\nNONTERMINALS:\n"
        for rule in self.rules:
            res += repr(rule) + "\n"
        return res

    def add_unit(self, left, term):
        self.terms.append(Unit(left, term))

    def add_rule(self, left, terms):
        self.rules.append(Rule(left, terms))
        return left


def init_grammar(file):
    grammar = Grammar()
    with open(file) as f:
        line = f.readline()
        terms = line.strip().split()
        for term in terms:
            grammar.nterms[term] = len(grammar.nterms)
        line = f.readline()
        grammar.start = line.strip()
        for line in f:
            terms = line.strip().split()
            i = 2
            while i < len(terms):
                j = i
                rhs = []
                while j < len(terms) and terms[j] != "|":
                    rhs.append(terms[j])
                    j += 1
                if len(rhs) == 1 and rhs[0] not in grammar.nterms:
                    grammar.add_unit(terms[0], rhs[0])
                else:
                    grammar.add_rule(terms[0], rhs)
                i = j + 1
    return grammar


def pretty_print(parse_tree):
    for slice in parse_tree:
        for list in slice:
            print(list)
        print("")


def cyk(string, grammar, debug=False):
    r = len(grammar)
    n = len(string)
    parse_tree = [[[None for k in range(0, r)] for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for unit in grammar.terms:
            # Terminals can now be matched by REGEX
            if re.match(unit.term, string[i]):
                parse_tree[i][i][grammar.nterms[unit.left]] = [[], unit]
    for l in range(0, n):
        for i in range(0, n-l):
            j = i + l
            for rule in grammar.rules:
                if len(rule.terms) == 1:
                    if parse_tree[i][j][grammar.nterms[rule.terms[0]]]:
                        parse_tree[i][j][grammar.nterms[rule.left]] = [[], rule]
                elif len(rule.terms) == 2:
                    for k in range(i, j):
                        if parse_tree[i][k][grammar.nterms[rule.terms[0]]] and parse_tree[k + 1][j][grammar.nterms[rule.terms[1]]]:
                            parse_tree[i][j][grammar.nterms[rule.left]] = [[k], rule]
                else:
                    for k in range(i, j-1):
                        for m in range(k, j):
                            if parse_tree[i][k][grammar.nterms[rule.terms[0]]] and parse_tree[k+1][m][grammar.nterms[rule.terms[1]]] and parse_tree[m+1][j][grammar.nterms[rule.terms[2]]]:
                                parse_tree[i][j][grammar.nterms[rule.left]] = [[k, m], rule]
    if debug:
        pretty_print(parse_tree)
    if parse_tree[0][n-1][grammar.nterms[grammar.start]]:
        return build_function(parse_tree, grammar, string, 0, n-1, grammar.start)
    return None


def build_function(parse_tree, grammar, input, i, j, nterm):
    r = grammar.nterms[nterm]
    # Find the rule chosen in the parse tree computation
    rule = parse_tree[i][j][r][1]

    # Renaming Operation or Terminal Translation
    if len(rule.terms) == 1:
        if rule.terms[0] == "VAR":
            return eq.Variable(input[i:j+1])
        elif rule.terms[0] == "CONST":
            return eq.Constant(float(input[i:j+1]))
        else:
            return build_function(parse_tree, grammar, input, i, j, rule.terms[0])

    # Binary Expansion of Term
    # Either unary operation or constant
    elif len(rule.terms) == 2:
        [k] = parse_tree[i][j][r][0]
        if rule.terms[0] == "SIN":
            return eq.Sine(build_function(parse_tree, grammar, input, k+1, j, rule.terms[1]))
        elif rule.terms[0] == "COS":
            return eq.Cosine(build_function(parse_tree, grammar, input, k+1, j, rule.terms[1]))
        elif rule.terms[0] == "LOG":
            return eq.Logarithm(build_function(parse_tree, grammar, input, k+1, j, rule.terms[1]))
    # Ternary Expansion of Term
    # Likely binary operation, check OP
    else:
        [k, m] = parse_tree[i][j][r][0]
        if rule.terms[1] == "ADD":
            return eq.Sum(build_function(parse_tree, grammar, input, i, k, rule.terms[0]), build_function(parse_tree, grammar, input, m+1, j, rule.terms[2]))
        elif rule.terms[1] == "SUB":
            return eq.Difference(build_function(parse_tree, grammar, input, i, k, rule.terms[0]), build_function(parse_tree, grammar, input, m+1, j, rule.terms[2]))
        elif rule.terms[1] == "MUL":
            return eq.Product(build_function(parse_tree, grammar, input, i, k, rule.terms[0]), build_function(parse_tree, grammar, input, m+1, j, rule.terms[2]))
        elif rule.terms[1] == "DIV":
            return eq.Quotient(build_function(parse_tree, grammar, input, i, k, rule.terms[0]), build_function(parse_tree, grammar, input, m+1, j, rule.terms[2]))
        elif rule.terms[1] == "MOD":
            return eq.Modulus(build_function(parse_tree, grammar, input, i, k, rule.terms[0]), build_function(parse_tree, grammar, input, m+1, j, rule.terms[2]))
        elif rule.terms[1] == "POW":
            return eq.Exponent(build_function(parse_tree, grammar, input, i, k, rule.terms[0]), build_function(parse_tree, grammar, input, m+1, j, rule.terms[2]))
        # ( EXPR ) / [ EXPR ] / { EXPR }
        elif rule.terms[1] == "EXPR":
            return build_function(parse_tree, grammar, input, k+1, m, rule.terms[1])


def parse():
    grammar = init_grammar(sys.argv[1])
    print(grammar)
    line = input("")
    while line != "quit":
        func = cyk(line, grammar)
        print(func)
        func.get_variables()
        print(func.variables)
        func.update_variable_indxs()
        print(func.evaluate([np.array([10, 11], dtype=np.float64), np.array([30, 32], dtype=np.float64)]))
        line = input("")


if __name__=='__main__':
    parse()