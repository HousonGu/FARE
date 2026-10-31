import ast, sys, types
from pathlib import Path
AUTHOR=Path(__file__).resolve().parents[2]/"third_party/PFRP"

def author_api():
    sys.path.insert(0,str(AUTHOR))
    # Isolate the author's package so another module named models cannot
    # shadow models.DLinear.
    package=types.ModuleType('_pfrp_author_models')
    package.__path__=[str(AUTHOR/'models')]
    sys.modules['_pfrp_author_models']=package
    # Remove the unused plotting import only; all forecasting classes/methods remain author code.
    path=AUTHOR/'models/DLinear_PFRP.py'; tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='utils.tools')]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=='models.DLinear':
            node.module='_pfrp_author_models.DLinear'
    api={'__name__':'pfrp_author_core'};exec(compile(tree,str(path),'exec'),api)
    cl=AUTHOR/'run_CL.py';tree=ast.parse(cl.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) or (isinstance(n,ast.Import) and all(a.name in ['torch','torch.nn','torch.nn.functional','numpy'] for a in n.names))]
    pcl={};exec(compile(tree,str(cl),'exec'),pcl)
    return api,pcl
