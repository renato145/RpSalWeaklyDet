from fastai.vision import *
from fastai.callbacks import *

__all__ = ['ActMode', 'get_class_weights', 'replace_layers', 'replace_layers_types', 'replace_activations', 'get_nchannels', 'format_db_labels']

ActMode = Enum('ActMode', 'Relu LeakyRelu')

def get_class_weights(data:DataBunch)->Tensor:
    '''
    Get weights to be used for calculating the loss when high class imbalance is present.
    Note: This calculation considers `data.classes[0]` as no class.
    '''
    counts = data.train_ds.inner_df.label.apply(lambda x: x[1]).explode().value_counts()
    n = counts.sum()
    for c in data.classes:
        if c not in counts: counts[c] = 0

    pos_weights = 1 - (counts / n)
    pos_weights = FloatTensor([pos_weights[o] for o in data.classes[1:]])[None]
    neg_weights = counts / n
    neg_weights = FloatTensor([neg_weights[o] for o in data.classes[1:]])[None]
    return pos_weights,neg_weights

def replace_layers(model:nn.Module, new_layer:Callable, func:Callable)->nn.Module:
    'Recursively replace layers in a model according to a condition `func`.'
    is_sequential = isinstance(model, nn.Sequential)
    it = enumerate(model.children()) if is_sequential else model.named_children()
    for name,layer in it:
        if func(layer):
            if is_sequential: model[name] = new_layer()
            else            : setattr(model, name, new_layer())

        replace_layers(layer, new_layer, func)

    return model

def replace_layers_types(model:nn.Module, new_layer:Callable, replace_types:Collection[nn.Module])->nn.Module:
    'Recursively replace layers in a model according to types `replace_types`.'
    def filter_types(layer): return any(isinstance(layer,o) for o in listify(replace_types))
    return replace_layers(model, new_layer=new_layer, func=filter_types)

def replace_activations(model:nn.Module, act_mode:ActMode)->nn.Module:
    if   act_mode == ActMode.Relu     : pass
    elif act_mode == ActMode.LeakyRelu: replace_layers_types(model, partial(nn.LeakyReLU, negative_slope=0.1, inplace=True), nn.ReLU)
    else: raise Exception(f'Invalid act_mode: {act_mode}')
    return model

def get_nchannels(m:nn.Module, layers:Collection[nn.Module], size:tuple=(64,64))->Collection[int]:
    with hook_outputs(layers) as hooks:
        x = dummy_batch(m, size)
        dummy_eval(m, size)

    return [o.stored.size(1) for o in hooks]

def format_db_labels(x:Tensor, n_classes:int)->Tensor:
    'Format the labels from `DataBunch` into a correct tensor.'
    return F.one_hot(x, n_classes).sum(1).clamp_max(1).float()[:,1:]
