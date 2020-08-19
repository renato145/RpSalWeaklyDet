from fastai.vision import *

def lse_pool1d(x:Tensor, r:float=1.0)->Tensor:
    n = x.size(-1)
    theta = tensor(n).float().to(x.device).log()
    return x.mul(r).logsumexp((-1)).sub(theta).div(r)

def lse_pool(x:Tensor, r:float=1.0)->Tensor:
    h,w = x.shape[-2:]
    theta = tensor(h*w).float().to(x.device).log()
    return x.mul(r).logsumexp((-2,-1)).sub(theta).div(r)

class LSELBAPool(Module):
    def __init__(self, r0:float=0.0):
        '''
        Log-Sum-Exp Pooling with Lower-bounded Adaptation (LSE-LBA Pool) (from https://arxiv.org/abs/1803.07703)

        :math:`\mathrm{p}=\mathrm{LSE}-\mathrm{LBA}(\mathrm{S})=\frac{1}{r_{0}+\exp (\beta)} \log \left\{\frac{1}{\mathrm{wh}} \sum_{i=1}^{\mathrm{w}} \sum_{j=1}^{\mathrm{h}} \exp \left[\left(r_{0}+\exp (\beta)\right) \mathrm{S}_{i, j}\right]\right\}`

        Refactored as:
        :math:`r = r_0 + \exp (\beta)`
        :math:`\theta =  \log(w\cdot h)`
        :math:`\mathrm{p}=\mathrm{LSE}-\mathrm{LBA}(\mathrm{S})=\frac{1}{r} \{\log [\sum_{i=1}^{\mathrm{w}} \sum_{j=1}^{\mathrm{h}} \exp (r \cdot \mathrm{S}_{i, j})] - \theta\}`
        '''
        self.r0 = r0
        self.beta = nn.Parameter(tensor([0.]))

    @property
    def r(self)->float:
        with torch.no_grad(): r = self.beta.exp().add(self.r0).item()
        return r

    def __repr__(self)->str: return f'{self.__class__.__name__} (r0={self.r0:.2f}, beta={self.beta.item():.4f} -> r={self.r:.4f})'

    def forward(self, x):
        r = self.beta.exp().add(self.r0)
        return lse_pool(x, r)
