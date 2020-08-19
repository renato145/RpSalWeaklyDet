from fastai.basics import *
from fastai.callbacks import *
from fastai.callbacks.hooks import _hook_inner

__all__ = ['MultiHook', 'MultiHooks', 'multi_hook_output', 'multi_hook_outputs', 'hook_input', 'hook_inputs']

class MultiHook(Hook):
    def __init__(self, m, hook_func, is_forward=True, detach=True):
        super().__init__(m, hook_func, is_forward=is_forward, detach=detach)
        self.stored = []

    def empty(self)->None: self.stored = []
    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored.append(self.hook_func(module, input, output))

class MultiHooks(Hooks):
    def __init__(self, ms:Collection[nn.Module], hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hooks = [MultiHook(m, hook_func, is_forward, detach) for m in ms]

    def empty(self)->None:
        for o in self.hooks: o.empty()

def multi_hook_output (module:nn.Module, detach:bool=True, grad:bool=False)->Hook:
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return MultiHook(module, _hook_inner, detach=detach, is_forward=not grad)

def multi_hook_outputs(modules:Collection[nn.Module], detach:bool=True, grad:bool=False)->Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return MultiHooks(modules, _hook_inner, detach=detach, is_forward=not grad)

def _hook_inner_input(m,i,o):
    if is_listy(i) and len(i)==1: i = i[0]
    return i if isinstance(i,Tensor) else i if is_listy(i) else list(i)

def hook_input (module:nn.Module, detach:bool=True, grad:bool=False)->Hook:
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner_input, detach=detach, is_forward=not grad)

def hook_inputs(modules:Collection[nn.Module], detach:bool=True, grad:bool=False)->Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner_input, detach=detach, is_forward=not grad)
