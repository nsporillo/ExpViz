import sys
import re


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
        self.units = []
        self.productions = []
        self.rules = set()
        self.terms2idx = dict()
        self.idx2terms = dict()
        self.nterms = dict()
        self.start = 0

    def __len__(self):
        return len(self.nterms)

    def __repr__(self):
        res = "["
        for unit in self.units:
            res += str(unit) + ", "
        for production in self.productions:
            res += str(production) + ", "
        for rule in self.rules:
            res += str(rule) + ", "
        res += "]"
        return res

    def add_unit(self, left, term):
        self.units.append(Unit(left, term))

    def add_production(self, left, A, B):
        self.productions.append(Production(left, A, B))

    def find_index(self, term):
        if term in self.terms2idx:
            return self.terms2idx[term]
        else:
            self.terms2idx[term] = len(self.terms2idx)
            self.idx2terms[self.terms2idx[term]] = term
            return self.terms2idx[term]

    def add_rule(self, left, terms):
        left = self.find_index(left)
        for i in range(0, len(terms)):
            terms[i] = self.find_index(terms[i])
        if left not in self.nterms:
            self.nterms[left] = self.idx2terms[left]
        self.rules.append(Rule(left, terms))
        return left


def expand_rule(grammar, rule):
    if len(rule.terms) == 1 and rule.terms[0] not in grammar.nterms:
        grammar.removed_rules.add(rule)
        for rule2 in grammar.rule[rule.terms[0]]:
            new = Rule(rule.left, rule2.terms)
            if new not in grammar.removed_rules:
                grammar.add_rule(rule.left, rule2.terms)



def cvt2cnf(grammar):
    # START
    grammar.start = grammar.add_rule("__START__", grammar.idx2terms[grammar.start])

    # TERM
    for rule in grammar.rules:
        for i in len(rule.terms):
            if rule.terms[i] not in grammar.nterms:
                rule.terms[i] = grammar.add_rule("$" + str(rule.terms[i]), rule.terms[i])

    # BIN
    for rule in grammar.rules:
        while len(rule.terms) > 2:
            A = rule.terms.pop(0)
            B = rule.terms.pop(0)
            rule.terms.insert(0, grammar.add_rule("$" + grammar.idx2terms[A] + grammar.idx2terms[B], [grammar.idx2terms[A], grammar.idx2terms[B]]))

    # DEL
    # NOTE, we won't have any epsilon rules, so we skip this here

    # UNIT
    for rule in grammar.rules:
        expand_rule(grammar, rule)

    return grammar


def init_grammar(file):
    grammar = Grammar()
    with open(file) as f:
        for line in f:
            terms = line.strip().split()
            i = 2
            while i < len(terms):
                j = i
                rhs = []
                while j < len(terms) and terms[j] != "|":
                    rhs.append(terms[j])
                    j += 1
                grammar.add_rule(terms[0], rhs)
                i = j + 1
    return grammar


def pretty_print(parse_tree):
    for slice in parse_tree:
        for list in slice:
            print(list)
        print("")


def cyk(input, grammar, debug=False):
    r = len(grammar)
    n = len(input)
    parse_tree = [[[None for k in range(0, r)] for j in range(0, n)] for i in range(0, n)]
    for i in range(0, n):
        for unit in grammar.units:
            # Terminals can now be matched by REGEX
            if re.match(unit.term, input[i]):
                    parse_tree[i][i][unit.left] = [-1, unit]
    for l in range(1, n):
        for i in range(0, n-l):
            j = i + l
            for k in range(i, j):
                for production in grammar.productions:
                    if parse_tree[i][k][production.A] and parse_tree[k+1][j][production.B]:
                        if not parse_tree[i][j][production.left] or parse_tree[i][j][production.left][0] > k:
                            parse_tree[i][j][production.left] = [k, production]
    if debug:
        pretty_print(parse_tree)
    if parse_tree[0][n-1][0]:
        print("Member: " + str(parse_tree[0][n-1][0]))
    else:
        print("Not a member")
    return parse_tree[0][n-1][0]


def parse():
    grammar = init_grammar(sys.argv[1])
    print(grammar)
    line = input("")
    while line != "quit":
        print(cyk(line, grammar, True))
        line = input("")

if __name__=='__main__':
    parse()