
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

def cyk(input, grammar):
    r = len(grammar)
    n = len(input)
    Parse = [[[False for i in range(0, r)] for j in range(0, n)] for k in range(0, n)]
    for s in range(0, n):
        for unit in grammar.units:
            if unit.val == input[s]:
                Parse[1, s, unit.idx] = True
    for l in range (1, n):
        for s in range(0, n-l+1):
            for p in range(0, l):
                for production in grammar.productions:
                    if Parse[p, s, production.first.idx] and Parse[l-p, s+p, production.second.idx]:
                        Parse[l, s, production.var.idx] = True
    if Parse[n, 1, 1]:
        return "Member"
    else:
        return "Not a memeber"

def parse():
    print("hello world")
    # TODO:
    # 1) Sort symbols left to right by left edge
    # 2)

if __name__=='__main__':
    parse()