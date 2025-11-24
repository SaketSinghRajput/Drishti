import ast, sys
p=r'f:\iit\scripts\compare_models.py'
try:
    s=open(p,'r',encoding='utf-8').read()
    ast.parse(s)
    print('AST_OK')
except Exception as e:
    print('PARSE_ERR', repr(e))
    import traceback
    traceback.print_exc()
