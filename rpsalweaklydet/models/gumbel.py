from fastai.vision import *
from ..utils import *

__all__ = ['gumbel_softmax', 'topk_anchors', 'GumbelPick']

def gumbel_softmax(logits:Tensor, factor:float=1.0, tau:float=1.0, hard:bool=False, dim:int=-1, reverse:bool=False)->Tensor:
    gumbels = -torch.empty_like(logits).exponential_().log().mul(factor)  # ~Gumbel(0,1)*`factor`
    gumbels = ( (-logits if reverse else logits) + gumbels ) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    return ret

def topk_anchors(anchors:Tensor, scores:Tensor, k:Optional[int]=None, reverse:bool=False)->Tuple[Tensor,Tensor]:
    k = ifnone(k, scores.size(0))
    idxs = scores.argsort(0, descending=not reverse)
    top_anchors = torch.stack([anchors[idxs[:k,i],i] for i in range(scores.size(1))], dim=1)
    top_scores = torch.stack([scores[idxs[:k,i],i] for i in range(scores.size(1))], dim=1)
    return top_anchors, top_scores

class GumbelPick(Module):
    def __init__(self, n_samples:int, softmax_dim:int=0, factor:float=1.0, reverse:bool=False):
        self.n_samples,self.softmax_dim,self.factor,self.reverse = n_samples,softmax_dim,factor,reverse
        self.sample_on_eval = False

    def __repr__(self)->str: return (f'{self.__class__.__name__}(n_samples={self.n_samples}, softmax_dim={self.softmax_dim}, ' +
                                     f'sample_on_eval={self.sample_on_eval}, factor={self.factor}, reverse={self.reverse})')

    def forward(self, anchors:Collection[Tensor], scores:Collection[Tensor])->Tensor:
        '''
        Input:
         - anchors: [n_anchors,n_classes,4] * bs
         - scores : [n_anchors,n_classes] * bs
        Output:
         - [bs, n_samples, n_classes, 4]
        '''
        bs = len(anchors)
        out = []

        for sample_anchors,sample_scores in zip(anchors,scores):
            this_scores = sample_scores.clone() # need to clone in order to not modify scores
            if self.training or self.sample_on_eval:
                o = []
                for _ in range(self.n_samples):
                    # TODO: check if works on multiple classes -> n_classes = scores[0].size(1), and return scores also
                    #       maybe put in a separate function.
                    # import pdb; pdb.set_trace()
                    sm = gumbel_softmax(this_scores, factor=self.factor, hard=True, dim=self.softmax_dim, reverse=self.reverse)
                    o.append(sample_anchors.mul(sm.unsqueeze(-1)).sum(dim=self.softmax_dim))
                    for i,j in zip(*torch.where(sm==1)):
                        this_scores[i,j].fill_(999. if self.reverse else -999) # Discard already picked regions

                o = torch.stack(o, dim=0)
            else:
                o,_ = topk_anchors(sample_anchors, sample_scores, k=self.n_samples, reverse=self.reverse)

            out.append(o)

        return torch.stack(out, dim=0)
