"""Load author modules in an isolated namespace, without changing methods."""
import ast,importlib.util,sys,types
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
AUTHOR=HERE.parents[1]/'third_party'/'SARAF'
def author_api():
    package='_saraf_author'
    if package+'.model' not in sys.modules:
        root=types.ModuleType(package);root.__path__=[str(AUTHOR)];sys.modules[package]=root
        spec=importlib.util.spec_from_file_location(package+'.retrieval',AUTHOR/'layers/Retrieval.py')
        retrieval=importlib.util.module_from_spec(spec);sys.modules[spec.name]=retrieval;spec.loader.exec_module(retrieval)
        path=AUTHOR/'models/SARAF.py';tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='layers.Retrieval':node.module=package+'.retrieval'
        model=types.ModuleType(package+'.model');exec(compile(tree,str(path),'exec'),model.__dict__);sys.modules[model.__name__]=model
    return sys.modules[package+'.model'].Model,sys.modules[package+'.retrieval'].RetrievalTool

def construct(c,device,control=False):
    Model,_=author_api()
    gpu=torch.device(device).index or 0
    config=SimpleNamespace(task_name='long_term_forecast',seq_len=56,pred_len=7,features='S',enc_in=1,n_period=1,topm=c['topm'],freq='d',time_aware_weight=c['time_aware_weight'],fusion_method='weighted',use_gpu=device.startswith('cuda'),gpu=gpu)
    model=Model(config).to(device)
    if control:return LinearControl(model.linear_x).to(device)
    return RetailModel(model).to(device)

class RetailModel(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x,retrieved):
        if x.ndim!=2 or x.shape[1]!=56 or retrieved.shape!=(len(x),7):raise ValueError('SARAF retail expects Bx56 history and Bx7 retrieval residual')
        # Author tensor layout G,B,P,C; only current batch kept on device.
        self.model.retrieval_temporal_dict={'retail':retrieved[None,:,:,None]}
        return self.model(x[:,:,None],None,torch.arange(len(x),device=x.device),mode='retail')[:,:,0]

class LinearControl(nn.Module):
    def __init__(self,linear):super().__init__();self.linear_x=linear
    def forward(self,x,retrieved):
        offset=x[:,-1:].detach();return self.linear_x(x-offset)+offset
