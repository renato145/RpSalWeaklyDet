import inspect
from fastai.vision import *

__all__ = ['Ints', 'delegates', 'get_percentile', 'normalize_tensor']

Ints = Union[int,Collection[int]]

def delegates(to=None, keep=False):
    'Decorator: replace `**kwargs` in signature with params from `to` (from: https://www.fast.ai/2019/08/06/delegation)'
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

def get_percentile(x:Tensor, perc:float=0.75)->Rank0Tensor:
    x = x.view(-1)
    n = x.numel()
    return x.kthvalue(int(round(n*perc)))[0]

def normalize_tensor(x: Tensor)->Tensor:
    x = x - x.min()
    return x/x.max()
