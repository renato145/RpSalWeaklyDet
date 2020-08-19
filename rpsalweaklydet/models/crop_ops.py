from fastai.vision import *
from torchvision.ops import roi_align, boxes as box_ops, nms
from torchvision.models.detection import _utils as det_utils
from .lse_pool import *
from ..utils import *

__all__ = ['GetVACrops', 'RoIAlign', 'RoiFilter', 'RoiFilterMulti', 'ApplyDeltas']

def _get_va_locs(anchors:Tensor, sz:int)->Tensor:
    'From x1,y1,x2,y2 to x,y,1,w,h'
    anchors = anchors / sz
    w = anchors[:,2]-anchors[:,0]
    h = anchors[:,3]-anchors[:,1]
    x = anchors[:,0].add(w.div(2)).mul(2).add(-1)
    y = anchors[:,1].add(h.div(2)).mul(2).add(-1)
    return torch.stack([x,y,torch.ones_like(x),w,h], dim=1)

@torch.jit.script
def _get_va_crop(x, N, loc):
    'Getting glimpse as in https://link.springer.com/chapter/10.1007/978-3-030-00934-2_15'
    # type: (Tensor, int, Tensor) -> Tensor
    dev = x.device
    bs,c,h,w = x.shape
    gx,gy,gs,gtx,gty = loc[:,0,None],loc[:,1,None],loc[:,2,None],loc[:,3,None],loc[:,4,None]
    ggx = (w+1) / 2*(gx+1)
    ggy = (h+1) / 2*(gy+1)
    sigma2 = abs(gs).view(-1,1,1)
    stridex = (max(h,w)-1) / (N-1) * abs(gtx)
    stridey = (max(h,w)-1) / (N-1) * abs(gty)
    grid_i = torch.arange(N, dtype=torch.float32).view(1,-1).to(dev)
    mu_x = ggx + (grid_i - N/2 - 0.5) * stridex
    mu_y = ggy + (grid_i - N/2 - 0.5) * stridey
    mu_x = mu_x.view(-1,N,1)
    mu_y = mu_y.view(-1,N,1)
    a = torch.arange(w, dtype=torch.float32).view(1,1,-1).to(dev)
    b = torch.arange(h, dtype=torch.float32).view(1,1,-1).to(dev)
    Fx = torch.exp(-(a-mu_x)**2 / (2*sigma2))
    Fy = torch.exp(-(b-mu_y)**2 / (2*sigma2))
    Fx = Fx / Fx.sum(2, keepdim=True).clamp_min(1e-8)
    Fy = Fy / Fy.sum(2, keepdim=True).clamp_min(1e-8)
    Fxt = Fx.transpose(2,1)
    glimpse = torch.stack([Fy@x[:,0]@Fxt, Fy@x[:,1]@Fxt, Fy@x[:,2]@Fxt], dim=1)
    return glimpse

class GetVACrops(Module):
    def __init__(self, crop_size:int):
        self.crop_size = crop_size

    def __repr__(self)->str: return f'{self.__class__.__name__}(crop_size={self.crop_size})'

    def forward(self, x:Tensor, anchors:Tensor)->Tensor:
        '''
        Input:
          - anchors: [bs,n_samples,n_classes,4]
        Output:
          - crops: [bs,n_samples,n_classes,3,crop_size,crop_size]
        '''
        sz = x.size(-1)
        n_samples,n_classes = anchors.size(1),anchors.size(2)
        out = []
        for i in range(n_samples):
            class_crops = []
            for j in range(n_classes):
                va_locs = _get_va_locs(anchors[:,i,j], sz)
                class_crops.append(_get_va_crop(x, self.crop_size, va_locs))

            out.append(torch.stack(class_crops, dim=1))

        return torch.stack(out, dim=1)

class RoIAlign(Module):
    def __init__(self, size:int, sampling_ratio:int=2):
        self.size,self.sampling_ratio = size,sampling_ratio

    def __repr__(self)->str: return f'{self.__class__.__name__}(size={self.size}, sampling_ratio={self.sampling_ratio})'
    def forward(self, x:Tensor, anchors:Collection[Tensor], input_size:int)->Tensor:
        scale = x.size(-1) / input_size
        return roi_align(x, anchors, (self.size,self.size), spatial_scale=scale, sampling_ratio=self.sampling_ratio)

class RoiFilter(Module):
    def __repr__(self)->str: return f'{self.__class__.__name__}()'
    def forward(self, anchors:Collection[Tensor], saliency_rois:Tensor, anchor_fts_rois:Tensor)->Tuple[Collection[Tensor],Ints,Tensor]:
        out_anchors,out_anchor_fts_rois = [],[]
        anchor_sizes = [o.size(0) for o in anchors]
        for anch,sal_rois,anch_fts in zip(anchors, saliency_rois.split(anchor_sizes), anchor_fts_rois.split(anchor_sizes)):
            scores = lse_pool(sal_rois, r=1).squeeze()
            out_anchors.append(anch)
            out_anchor_fts_rois.append(anch_fts.mul(sal_rois.sigmoid()))

        anchor_sizes = [o.size(0) for o in out_anchors]
        anchor_fts_rois = torch.cat(out_anchor_fts_rois, dim=0)
        return out_anchors,anchor_sizes,anchor_fts_rois

class RoiFilterMulti(Module):
    def __init__(self):
        'Multiclass version of `RoiFilter`.'
        self.roi_filter = RoiFilter()

    def __repr__(self)->str: return f'{self.__class__.__name__}({self.roi_filter})'
    def forward(self, anchors:Collection[Tensor], saliency_rois:Tensor, anchor_fts_rois:Collection[Tensor]
               )->Collection[Tuple[Collection[Tensor],Ints,Tensor]]:
        out_anchors,out_anchors_sizes,out_anchor_fts_rois = [],[],[]
        for i,o in enumerate(anchor_fts_rois):
            a,b,c = self.roi_filter(anchors, saliency_rois[:,i,None], o)
            out_anchors.append(a)
            out_anchors_sizes.append(b)
            out_anchor_fts_rois.append(c)

        return out_anchors,out_anchors_sizes,out_anchor_fts_rois

class ApplyDeltas(Module):
    def __init__(self):
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def process_one(self, deltas:Tensor, anchors:Tensor, size:int)->Tensor:
        anchors = self.box_coder.decode(deltas, [anchors])
        anchors = box_ops.clip_boxes_to_image(anchors, (size,size))
        return anchors

    def forward(self, picked_deltas:Collection[Tensor], picked_anchors:Collection[Tensor], size:int)->Collection[Tensor]:
        return [self.process_one(deltas,anchors,size) for deltas,anchors in zip(picked_deltas,picked_anchors)]
