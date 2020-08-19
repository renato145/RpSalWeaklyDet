from fastai.vision import *
from fastai.callbacks import *
from torchvision.models.detection.faster_rcnn import AnchorGenerator
from torchvision.ops import boxes as box_ops
from ..utils import *
import torchvision

__all__ = ['SimpleAnchorGenerator']

class DetImageList():
    def __init__(self, xb:Tensor):
        self.xb = xb
        self.tensors = xb
        self.image_sizes = [x.shape[-2:] for x in xb]

def _generate_anchors(input_size:int, fts_size:int, anchor_sizes:Ints, aspect_ratios:Floats, stride:int)->Tensor:
    scales = tensor(anchor_sizes).float()
    aspect_ratios = tensor(aspect_ratios).float()
    h_ratios = aspect_ratios.sqrt()
    w_ratios = 1/h_ratios
    ws = (w_ratios[:,None] * scales[None,:]).view(-1)
    hs = (h_ratios[:,None] * scales[None,:]).view(-1)
    base_anchors = torch.stack([-ws,-hs,ws,hs], dim=1).div(2).round()
    grid_size = fts_size / stride
    strides = input_size / grid_size
    shifts = torch.arange(grid_size+1).mul(strides).float()
    sy,sx = torch.meshgrid(shifts,shifts)
    sy,sx = sy.reshape(-1),sx.reshape(-1)
    shifts = torch.stack([sx,sy,sx,sy], dim=1)
    return shifts.view(-1,1,4).add(base_anchors.view(1,-1,4)).reshape(-1,4)

class SimpleAnchorGenerator(Module):
    def __init__(self, input_size:int, anchor_sizes:Ints=(64,128,256), aspect_ratios:Floats=(1.0,), strides:Ints=[8,16,32]):
        'Simple anchor generator, generates all posible anchors for an image [x1,y1,x2,y2].'
        anchor_sizes,aspect_ratios = listify(anchor_sizes),listify(aspect_ratios)
        strides = listify(strides, len(anchor_sizes))
        self.input_size,self.anchor_sizes,self.aspect_ratios,self.strides = input_size,anchor_sizes,aspect_ratios,strides
        self.anchors = None

    def __repr__(self)->str:
        return (f'{self.__class__.__name__}(input_size={self.input_size}, anchor_sizes={self.anchor_sizes}, '+
                                           f'aspect_ratios={self.aspect_ratios}, strides={self.strides})')

    def generate_anchors(self, x:Tensor)->None:
        anchors = torch.cat([_generate_anchors(self.input_size, x.size(-1), listify(anchor_sizes), self.aspect_ratios, stride)
                             for anchor_sizes,stride in zip(self.anchor_sizes,self.strides)], dim=0)

        # Filter anchors
        anchors = box_ops.clip_boxes_to_image(anchors, (self.input_size,self.input_size))
        keep = box_ops.remove_small_boxes(anchors, 1e-3)
        self.anchors = anchors[keep]

    def forward(self, x):
        if self.anchors is None: self.generate_anchors(x)
        return [self.anchors.clone().to(x.device) for _ in range(x.size(0))]
