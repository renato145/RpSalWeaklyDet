from fastai.vision import *
from fastai.callbacks import *
from .loss_func import SaliencyMultiBCE
from .lse_pool import LSELBAPool
from .utils import *
from ..utils import *
from ..metrics import *

HeadMode = Enum('HeadMode', 'Resnet Densenet')

_head_mode = {
    models.resnet18 :HeadMode.Resnet, models.resnet34: HeadMode.Resnet,
    models.resnet50 :HeadMode.Resnet, models.resnet101:HeadMode.Resnet,
    models.resnet152:HeadMode.Resnet,

    models.densenet121:HeadMode.Densenet, models.densenet169:HeadMode.Densenet,
    models.densenet201:HeadMode.Densenet, models.densenet161:HeadMode.Densenet,
}

def bn_act_conv(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None,
                activ:Optional[Callable]=partial(nn.ReLU, inplace=True), init:Callable=nn.init.kaiming_normal_):
    if padding is None: padding = (ks-1)//2
    conv = init_default(nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=padding), init)
    return nn.Sequential(nn.BatchNorm2d(ni), activ(), conv)


class MergeMaps(Module):
    'Concatenate saliency maps'
    def __init__(self, dim=1): self.dim = dim
    def __repr__(self)->str: return f'{self.__class__.__name__}(dim={self.dim})'
    def forward(self, x:Collection[Tensor]): return torch.cat(x, dim=self.dim)

class MultiHeadBase(Module):
    def __init__(self, ni:int, nf:int, ks:int):
        self.convs1 = nn.ModuleList([bn_act_conv(ni, ni//2, ks=ks) for _ in range(nf)])
        self.convs2 = nn.ModuleList([bn_act_conv(ni//2,  1, ks=ks) for _ in range(nf)])
        self.merge = MergeMaps(dim=1)

    def forward(self, x): return self.merge([conv2(conv1(x)) for conv1,conv2 in zip(self.convs1,self.convs2)])

class MultiPool(Module):
    def __init__(self, c:int, r0:float=0.0):
        'Applies individual pooling to each channel of the saliency map.'
        self.pools = nn.ModuleList([LSELBAPool(r0=r0) for _ in range(c)])

    def forward(self, x):
        'input format: [bs,c,sz,sz]'
        return torch.cat([pool(x[:,i,None]) for i,pool in enumerate(self.pools)], dim=1)

class MultiFPNHead(Module):
    def __init__(self, body:nn.Module, c:int, mode:HeadMode, r0:float=0.0, ks:int=1, nf:int=256):
        self.c,self.nf = c,nf
        hook_layers = self._get_hook_layers(body, mode)
        nis = get_nchannels(body, hook_layers)
        self.hooks = hook_outputs(hook_layers, detach=False)
        self.inner_blocks = nn.ModuleList([nn.Conv2d(ni,nf,1) for ni in nis])
        # self.layer_blocks = nn.ModuleList([nn.Conv2d(nf,nf,3,padding=1) for _  in nis]) # Exclude for saliency map purposes
        self.conv = MultiHeadBase(nf, c, ks=ks)
        self.pool = MultiPool(c, r0=r0)
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_saliency_layer(self)->nn.Module: return self.conv

    def _get_hook_layers(self, m:nn.Module, mode:HeadMode)->Collection[nn.Module]:
        if   mode == HeadMode.Resnet  : layers = list(m.children())[-4:]
        elif mode == HeadMode.Densenet: layers = [m[0][i][1] for i in [5,7,9]] + [m[0][12]]
        return layers

    def forward(self, x):
        for i,(fts,ib) in enumerate(zip(self.hooks.stored[::-1], self.inner_blocks[::-1])):
            if i==0:
                last_inner = ib(fts)
            else:
                inner_lateral = ib(fts)
                inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='nearest')
                last_inner = inner_lateral + inner_top_down

        return self.pool(self.conv(last_inner))

    def __del__(self)->None: self.hooks.remove()


def get_classifier(arch:Callable, c:int, r0:float=0.0, pretrained:bool=True, act_mode:ActMode=ActMode.LeakyRelu, **kwargs:Any)->nn.Module:
    'Build the model for classification.'
    body = create_body(arch, pretrained=pretrained)
    head_mode = _head_mode[arch]
    if head_mode == HeadMode.Densenet: body[0].add_module('final_relu', nn.ReLU(True))
    body = replace_activations(body, act_mode)

    head = MultiFPNHead(body, c=c, mode=head_mode, r0=r0, **kwargs)
    head = replace_activations(head, act_mode)

    return nn.Sequential(body, head)

@delegates(Learner.__init__)
def get_classifier_learner(data:DataBunch, arch:Callable, act_mode:ActMode=ActMode.LeakyRelu, pretrained:bool=True,
                           model_kwargs:Optional[dict]=None, **kwargs:Any)->Learner:
    from fastai.vision.learner import cnn_config

    meta = cnn_config(arch)
    model_kwargs = ifnone(model_kwargs, dict())
    model = get_classifier(arch=arch, c=data.c-1, pretrained=pretrained, act_mode=act_mode, **model_kwargs)
    saliency_layer = model[1].get_saliency_layer()
    opt_func = partial(optim.Adam, eps=0.1, betas=(0.9,0.99))

    # Set Metrics
    metrics = []
    labels = data.classes[1:]

    def auc_pred_proc(x, i): return x[:,i]
    def auc_targ_proc(x, i): return format_db_labels(x[1], n_classes=data.c)[:,i]

    auc_metrics = [BinaryAUROC(f'auc_{lbl}', do_sigmoid=True,
                               pred_post_proc=partial(auc_pred_proc, i=i),
                               targ_post_proc=partial(auc_targ_proc, i=i))
                   for i,lbl in enumerate(labels)]
    metrics += auc_metrics
    metrics.append(MeanAUROC(auc_metrics))


    learn = Learner(data, model, metrics=metrics, opt_func=opt_func, wd=1e-3, **kwargs)
    learn.split(meta['split'])
    if pretrained: learn.freeze()
    apply_init(model[1], nn.init.kaiming_normal_)

    # Loss func
    pos_weights,neg_weights = get_class_weights(data)
    loss_scale = 10.0
    learn.loss_func = SaliencyMultiBCE(pos_weights=pos_weights, neg_weights=neg_weights, loss_scale=loss_scale).to(data.device)

    return learn
