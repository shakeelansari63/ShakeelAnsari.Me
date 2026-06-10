# PLY — Lex & Yacc for Python

PLY is a Python implementation of Lex and Yacc. Install with `pip install ply`.

## Lexer — Tokenizing

The lexer breaks input into tokens using regex rules.

```python
from ply import lex

# Define token names
tokens = ("plus", "minus", "multiply", "divide", "lpar", "rpar", "number")

# Regex rules for each token: t_<TOKEN_NAME>
t_plus = r'\+'
t_minus = r'\-'
t_multiply = r'\*'
t_divide = r'/'
t_lpar = r'\('
t_rpar = r'\)'

# For tokens that need extra processing, use a function
def t_number(t):
    r'\d+'
    t.value = int(t.value)
    return t

# Track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# Ignore whitespace
def t_whitespace(t):
    r'[\s\t]+'

# Error handling
def t_error(t):
    print(f'Illegal Character: {t.value}')
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()
```

### Using the Lexer

```python
data = '3 +(4-7)*2'
lexer.input(data)

tok = lexer.token()
while tok:
    print(tok)
    tok = lexer.token()
```

## Yacc — Parsing

The parser defines grammar rules and builds a parse tree.

```python
from ply import yacc
from p_lex import tokens

# Grammar rules: p_<rule_name>
# The docstring contains the grammar in BNF-like form.
def p_expressions(p):
    '''expr : expr plus expr
            | expr minus expr
            | expr multiply expr
            | expr divide expr
    '''
    if p[2] == '+':
        p[0] = p[1] + p[3]
    elif p[2] == '-':
        p[0] = p[1] - p[3]
    elif p[2] == '*':
        p[0] = p[1] * p[3]
    elif p[2] == '/':
        p[0] = p[1] / p[3]

def p_expr_in_paranthesis(p):
    'expr : lpar expr rpar'
    p[0] = p[2]

def p_number_expr(p):
    'expr : number'
    p[0] = p[1]

def p_error(p):
    print("Syntax error!")

parser = yacc.yacc()
```

### Using the Parser

```python
inp_expr = input("Give Expression: ")
result = parser.parse(inp_expr)
print(result)
```

## How It Works

1. **Tokens** are named symbols (e.g., `plus`, `number`).
2. **Lexer rules** use regex to match tokens from input text.
3. **Grammar rules** define how tokens combine into expressions.
4. `p[0]` holds the result of the rule; `p[1]`, `p[2]`, etc. are the components.
5. Precedence is handled by rule order — rules listed first have higher precedence.
