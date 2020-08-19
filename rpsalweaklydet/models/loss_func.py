from fastai.vision import *
from .utils import format_db_labels

def saliency_multi_bce(input:Tensor, target:Tensor, pos_weights:Tensor, neg_weights:Tensor)->Rank0Tensor:
    # Original code before optimize it:
    # preds = input.sigmoid()
    # loss = (-preds.log().mul(target).mul(pos_weights) - (1-preds).log().mul(1-target).mul(neg_weights)).mean()
    preds = F.logsigmoid(input)
    return (-preds.mul(target).mul(pos_weights) - preds.sub(input).mul(1-target).mul(neg_weights)).mean()

class ParallelLossMetrics(callbacks.LossMetrics):
    'Modification for Parallel models'
    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        bs = last_target[0].size(0)
        for name in self.names:
            self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

class SaliencyMultiBCE(Module):
    def __init__(self, pos_weights:Optional[Tensor]=None, neg_weights:Optional[Tensor]=None, loss_scale:float=10.0):
        '''
        Parameters:
        - loss_scale: multiply the final loss, as the pos and neg weights may produce very low values.
        '''
        self.pos_weights = nn.Parameter(pos_weights, requires_grad=False)
        self.neg_weights = nn.Parameter(neg_weights, requires_grad=False)
        self.c = pos_weights.size(1)
        self.loss_scale = nn.Parameter(tensor(loss_scale).float(), requires_grad=False)

    def forward(self, input:Tensor, targs_bbs:Tensor, targs_lbls:Tensor)->Tensor:
        lbls = format_db_labels(targs_lbls, n_classes=self.c+1) # +1 because of the No class label
        loss = saliency_multi_bce(input, lbls, pos_weights=self.pos_weights, neg_weights=self.neg_weights)
        return loss.mul(self.loss_scale)

class SaliencyDetMultiBCE(SaliencyMultiBCE):
    def __init__(self, pos_weights:Optional[Tensor]=None, neg_weights:Optional[Tensor]=None, loss_scale:float=10.0,
                 loss_weights:Optional[Floats]=None):
        '''
        Parameters:
        - loss_scale: multiply the final loss, as the pos and neg weights may produce very low values.
        - loss_weights: apply different weights to full image and crop losses.
        '''
        super().__init__(pos_weights=pos_weights, neg_weights=neg_weights, loss_scale=loss_scale)
        self.metric_names = ['bce_crop','bce_full']
        n = len(self.metric_names)
        loss_weights = ifnone(loss_weights, listify(1, n))
        assert len(loss_weights) == n, f'`loss_weights` should have {n} elements: {loss_weights}'
        self.loss_weights = nn.Parameter(tensor(loss_weights).float(), requires_grad=False)

    def forward(self, input:Tuple[Tensor,Tensor], targs_bbs:Tensor, targs_lbls:Tensor)->Tensor:
        'input = ([bbs],[lbls])'
        lbls = format_db_labels(targs_lbls, n_classes=self.c+1) # +1 because of the No class label
        losses = [saliency_multi_bce(input[1][:,i], lbls, pos_weights=self.pos_weights, neg_weights=self.neg_weights).mul(w)
                  for i,w in enumerate(self.loss_weights[:2])]
        self.metrics = dict(zip(self.metric_names, losses))
        return sum(losses)
