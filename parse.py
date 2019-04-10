import sys
import re

"""
Grammar:
    S -> EXPR
    EXPR -> EXPR OP EXPR | VAR | CONST
    OP -> + | - | x | /
    VAR -> A-Z,a-z
    CONST -> DIGIT NUM DOT NUM | NUM
    NUM -> DIGIT NUM | DIGIT
    DIGIT -> 0-9
    DOT -> .

CNF Grammar:
    S -> EXPR+ EXPR | A-Z,a-z | NUM+ DEC | DIGIT NUM | 0-9
    EXPR -> EXPR+ EXPR | A-Z,a-z | NUM+ DEC | DIGIT NUM | 0-9
    EXPR+ -> EXPR OP
    OP -> + | - | x | /
    VAR -> A-Z,a-z
    CONST -> NUM+ DEC | DIGIT NUM | 0-9
    NUM+ -> DIGIT NUM
    DEC -> DOT NUM
    NUM -> DIGIT NUM | 0-9
    DIGIT -> 0-9
    DOT -> .
"""


class Unit:
    def __init__(self, left, term):
        self.left = left
        self.term = term

    def __str__(self):
        return str(self.left) + " -> " + self.term


class Production:
    def __init__(self, left, first, second):
        self.left = left
        self.A = first
        self.B = second

    def __str__(self):
        return str(self.left) + " -> " + str(self.A) + " " + str(self.B)


class Grammar:
    def __init__(self):
        self.units = []
        self.productions = []
        self.nterms = dict()

    def __len__(self):
        return len(self.units) + len(self.productions)

    def __str__(self):
        res = "["
        for unit in self.units:
            res += str(unit) + ", "
        for production in self.productions:
            res += str(production) + ", "
        res += "]"
        return res

    def add_unit(self, left, term):
        self.units.append(Unit(left, term))

    def add_production(self, left, first, second):
        self.productions.append(Production(left, first, second))


def init_grammar(file):
    grammar = Grammar()
    nonterms = grammar.nterms
    min_idx = 0
    with open(file) as f:
        for line in f:
            terms = line.strip().split()
            i = 2
            if terms[0] in nonterms:
                idx = nonterms[terms[0]]
            else:
                idx = min_idx
                min_idx += 1
                nonterms[terms[0]] = idx
            while i < len(terms):
                if i+1 == len(terms) or terms[i+1] == "|":
                    grammar.add_unit(idx, terms[i])
                    i = i + 2
                else:
                    A = terms[i]
                    B = terms[i+1]
                    if A in nonterms:
                        A = nonterms[A]
                    else:
                        nonterms[A] = min_idx
                        A = min_idx
                        min_idx += 1
                    if B in nonterms:
                        B = nonterms[B]
                    else:
                        nonterms[B] = min_idx
                        B = min_idx
                        min_idx += 1
                    grammar.add_production(idx, A, B)
                    i = i + 3
    return grammar


def pretty_print(parse_tree):
    for slice in parse_tree:
        for list in slice:
            print(list)
        print("")

def cyk(input, grammar):
    r = len(grammar)
    n = len(input)
    parse_tree = [[[False for k in range(0, r)] for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for unit in grammar.units:
            # Terminals can now be matched by REGEX
            if re.match(unit.term, input[i]):
                parse_tree[i][i][unit.left] = True
    for l in range(1, n):
        for i in range(0, n-l):
            j = i + l
            for k in range(i, j):
                for production in grammar.productions:
                    if parse_tree[i][k][production.A] and parse_tree[k+1][j][production.B]:
                        parse_tree[i][j][production.left] = True
    pretty_print(parse_tree)
    if parse_tree[0][n-1][0]:
        return "Member"
    else:
        return "Not a member"


def parse():
    grammar = init_grammar(sys.argv[1])
    print(grammar)
    line = input("")
    while line != "quit":
        print(cyk(line, grammar))
        line = input("")

if __name__=='__main__':
    parse()