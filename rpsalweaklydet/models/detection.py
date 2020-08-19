from fastai.vision import *
from fastai.callbacks import *
from .classifier import get_classifier
from .anchor_generator import SimpleAnchorGenerator
from .loss_func import SaliencyDetMultiBCE, ParallelLossMetrics
from .gumbel import *
from .crop_ops import *
from .utils import *
from ..utils import *
from ..structures import *
from ..metrics import *

class SplitAnchors(nn.Module):
    def forward(self, x:Tensor, anchor_sizes:Collection[int])->Tensor: return x.split(anchor_sizes)

class SingleHead(Module):
    def __init__(self, ni:int, det_nf:int=1, reg_nf:int=4):
        detection = nn.Linear(ni,det_nf, bias=False)
        regression  = nn.Linear(ni,reg_nf)
        torch.nn.init.constant_(regression.weight, 0)
        torch.nn.init.constant_(regression.bias  , 0)
        torch.nn.init.normal_(detection.weight, mean=0.01, std=0.01)
        # torch.nn.init.constant_(detection.bias, 1e-3) # Try smaller 1e-5
        self.detection  = detection
        self.regression = regression
        self.detection_split = SplitAnchors()
        self.regression_split = SplitAnchors()

    def forward(self, x:Tensor, anchor_sizes:Ints)->Tuple[Tensor,Tensor]:
        det,reg = self.detection(x),self.regression(x)
        return self.detection_split(det, anchor_sizes),self.regression_split(reg, anchor_sizes)

def _get_intermediate_layers(ni:int, nf:int)->nn.Module:
    return nn.Sequential(conv_layer(ni, nf, ks=3, norm_type=None), nn.AdaptiveMaxPool2d(1), Flatten())

class HeadDetection(Module):
    def __init__(self, ni:int, nf:int, det_nf:int=1, reg_nf:int=4):
        'Uses the same layers to process all classses.'
        self.int_layers = _get_intermediate_layers(ni, nf)
        self.head = SingleHead(nf, det_nf=det_nf, reg_nf=reg_nf)

    @property
    def scores_layer(self)->nn.Module: return self.head.detection_split
    @property
    def regression_layer(self)->nn.Module: return self.head.regression

    def forward(self, x:Collection[Tensor], anchor_sizes:Collection[Ints])->Tuple[Collection[Tensor],Collection[Tensor]]:
        out = [self.head(self.int_layers(o), sz) for o,sz in zip(x,anchor_sizes)]
        return list(zip(*out))

class SaliencyDetection(Module):
    def __init__(self, backbone:nn.Module, classifier:nn.Module, anchor_generator:nn.Module, c:int, n_samples:int=1,
                 crop_sz:int=64, anchor_ft_sz:int=5, int_nf:int=64, gumbel_factor:float=1.0):
        '''
        # TODO: finish specifying all parameters here...
        Parameters:
        '''
        self.backbone,self.anchor_generator,self.c,self.n_samples =\
             backbone,     anchor_generator,     c,     n_samples
        self.saliency_hook = hook_output(self.saliency_layer)
        self.anchors_hook = hook_outputs(self.anchor_layers, detach=False)
        self.roi_align = RoIAlign(anchor_ft_sz)
        self.roi_filter = RoiFilterMulti()
        ni = backbone[1].nf // 2
        self.head = HeadDetection(ni, int_nf)
        self.apply_deltas = ApplyDeltas()
        self.gumbel_pick = GumbelPick(n_samples=n_samples, softmax_dim=0, factor=gumbel_factor)
        self.get_crops = GetVACrops(crop_sz)

        self.classifier = classifier

    @property
    def saliency_layer(self)->nn.Module: return self.backbone[1].get_saliency_layer()
    @property
    def anchor_layers(self)->Collection[nn.Module]: return self.saliency_layer.convs1

    def get_anchor_fts(self)->Collection[Tensor]: return self.anchors_hook.stored

    def forward(self, x):
        bs,nf,sz = x.shape[:3]
        full_img_logits = self.backbone(x)
        saliency,anchor_fts = self.saliency_hook.stored,self.get_anchor_fts()
        anchors = self.anchor_generator(saliency)
        saliency_rois = self.roi_align(saliency, anchors, sz)
        anchor_fts_rois = [self.roi_align(o, anchors, sz) for o in anchor_fts]
        anchors,anchor_sizes,anchor_fts_rois = self.roi_filter(anchors, saliency_rois, anchor_fts_rois)
        # anchors         -> c * bs * Tensor[anchors,4]
        # anchor_sizes    -> c * bs * [size]
        # anchor_fts_rois -> c * Tensor[sum(sizes),crop_sz,anchor_ft_sz,anchor_ft_sz]
        det_scores,det_deltas = self.head(anchor_fts_rois, anchor_sizes)
        # det_scores -> c * bs * Tensor[anchors,1]
        # det_deltas -> c * bs * Tensor[anchors,4] 
        anchors = [self.apply_deltas(cdeltas, canchors, sz) for cdeltas,canchors in zip(det_deltas,anchors)]
        final_anchors = torch.cat([self.gumbel_pick(canchors, cdet_scores) for canchors,cdet_scores in zip(anchors,det_scores)], dim=2)
        # final_anchors -> [bs,n_samples,n_classes,4]
        crops = self.get_crops(x, final_anchors)
        # crops -> [bs,n_samples,n_classes,3,sz,sz]
        bb = format_bb(final_anchors, sz)
        # Get logits per sample

        crops_logits = self.classifier(crops)
        # crop_logits -> [bs,n_samples,n_classes]
        crops_logits = crops_logits.mean(dim=1)
        # crop_logits -> [bs,n_classes]
        crops_logits.add_(full_img_logits)
        preds = [crops_logits,full_img_logits]
        # preds -> [bs,2,n_classes]
        #              0 -> crop logit
        #              1 -> full image logit
        preds = torch.stack(preds, dim=1)
        return bb,preds

def _individual_head(nf:int)->nn.Module:
    layers = bn_drop_lin(nf, nf//2, bn=True, p=0.5, actn=nn.ReLU(inplace=True)) + bn_drop_lin(nf//2, 1, p=0.5)
    return nn.Sequential(*layers)

class CropClassifier(Module):
    'Uses the same body, but different heads to classify each crop class.'
    def __init__(self, model:nn.Module):
        # if classic_classifier: model,heads = _modify_classic_model(model)
        body = nn.Sequential(model[0], model[1][:2])
        nf = model[1][2].num_features
        c = model[1][-1].out_features
        heads = nn.ModuleList([_individual_head(nf) for _ in range(c)])
        self.model,self.heads = body,heads

    @property
    def layers2freeze(self)->Collection[nn.Module]:
        return flatten_model(self.model)

    def forward(self, x:Tensor):
        bs,n_samples,n_classes,_,sz,_ = x.shape
        x = x.view(-1,3,sz,sz)
        out =  self.model(x).view(bs*n_samples,n_classes,-1)
        out = torch.cat([head(out[:,i]) for i,head in enumerate(self.heads)], dim=-1).view(bs,n_samples,-1)
        return out

@delegates(SaliencyDetection.__init__)
def get_detection_model(arch:Callable, c:int, img_sz:int, crop_arch:Optional[Callable]=None, pretrained:bool=True,
                        crop_sz:int=64, anchor_sizes:Ints=(64,128,256), aspect_ratios:Floats=(1.0,), strides:Ints=[8,16,32],
                        act_mode:ActMode=ActMode.LeakyRelu, backbone_kwargs:Optional[dict]=None, **kwargs:Any)->nn.Module:
    crop_arch = ifnone(crop_arch, arch)
    backbone_kwargs = ifnone(backbone_kwargs, dict())
    backbone = get_classifier(arch=arch, c=c, act_mode=act_mode, pretrained=pretrained, **backbone_kwargs)
    anchor_generator = SimpleAnchorGenerator(img_sz, anchor_sizes, aspect_ratios, strides)
    classifier = CropClassifier(create_cnn_model(crop_arch, c))

    return SaliencyDetection(backbone, classifier, anchor_generator, c, crop_sz=crop_sz, **kwargs)

@delegates(Learner.__init__)
def get_detection_learner(data:DataBunch, arch:Callable, crop_arch:Optional[Callable]=None, pretrained:bool=True, crop_sz:int=64,
                          act_mode:ActMode=ActMode.LeakyRelu, 
                          backbone_kwargs:Optional[dict]=None, model_kwargs:Optional[dict]=None, **kwargs:Any)->Learner:
    img_sz = data.x[0].size[0]
    model_kwargs = ifnone(model_kwargs, dict())
    model = get_detection_model(arch=arch, c=data.c-1, img_sz=img_sz, crop_arch=crop_arch, pretrained=pretrained, crop_sz=crop_sz,
                                act_mode=act_mode, backbone_kwargs=backbone_kwargs, **model_kwargs)

    if pretrained:
        freeze_layers = flatten_model(model.backbone) + model.classifier.layers2freeze
        for l in freeze_layers:
            if not isinstance(l, bn_types): requires_grad(l, False)

    # Set Metrics
    metrics = []
    labels = data.classes[1:]

    def auc_pred_proc(x,i,lbl): return x[1][:,i,lbl]
    def auc_targ_proc(x,lbl): return format_db_labels(x[1], n_classes=data.c)[:,lbl]

    auc_metrics = [BinaryAUROC(f'auc_{lbl}', do_sigmoid=True,
                               pred_post_proc=partial(auc_pred_proc, i=1, lbl=idx),
                               targ_post_proc=partial(auc_targ_proc, lbl=idx))
                   for idx,lbl in enumerate(labels)]
    metrics += auc_metrics
    metrics.append(MeanAUROC(auc_metrics))

    # Get Learner
    opt_func = partial(optim.Adam, eps=0.1, betas=(0.9,0.99))
    learn = Learner(data, model, metrics=metrics, opt_func=opt_func, wd=1e-3, **kwargs)
    learn.split((learn.model.classifier,))

    # Loss func
    pos_weights,neg_weights = get_class_weights(data)
    loss_scale = 10.0
    learn.loss_func = SaliencyDetMultiBCE(pos_weights=pos_weights, neg_weights=neg_weights, loss_scale=loss_scale).to(data.device)
    learn.callback_fns.append(ParallelLossMetrics)

    return learn
