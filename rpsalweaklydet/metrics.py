from fastai.basics import *
from fastai.callbacks import *

__all__ = ['BinaryAUROC', 'MeanAUROC']

@dataclass
class BinaryAUROC(Callback):
    "Computes the area under the curve (AUC) score based on the receiver operator characteristic (ROC) curve. Restricted to binary classification tasks."
    def __init__(self, name:str='AUC', do_sigmoid:bool=False, pred_post_proc:Optional[Callable]=None,
                 targ_post_proc:Optional[Callable]=None):
        self.name,self.do_sigmoid,self.pred_post_proc,self.targ_post_proc = name,do_sigmoid,pred_post_proc,targ_post_proc

    def __repr__(self)->str: return f'{self.__class__.__name__}({self.name!r})'

    def on_epoch_begin(self, **kwargs:Any)->None:
        self.targs,self.preds,self.auroc = LongTensor([]),Tensor([]),tensor(0.0)

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs:Any)->None:
        if self.pred_post_proc: last_output = self.pred_post_proc(last_output)
        if self.do_sigmoid: last_output = last_output.sigmoid()
        if self.targ_post_proc: last_target = self.targ_post_proc(last_target)
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu().long()))

    def on_epoch_end(self, last_metrics:Tensor, **kwargs0:Any)->None:
        self.auroc = auc_roc_score(self.preds, self.targs)
        return add_metrics(last_metrics, self.auroc)

@dataclass
class MeanAUROC(Callback):
    _order = 10
    def __init__(self, auroc_metrics:Collection[BinaryAUROC], name:str='mean_auroc'):
        self.auroc_metrics,self.name = auroc_metrics,name

    def on_epoch_end(self, last_metrics:TensorOrNumList, **kwargs:Any)->dict:
        results = torch.stack([o.auroc for o in self.auroc_metrics])
        mean_auroc = results.mean() if len(results)>0 else 0
        return add_metrics(last_metrics, mean_auroc)
