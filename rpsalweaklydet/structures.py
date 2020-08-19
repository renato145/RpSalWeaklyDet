import cv2
import seaborn as sns
from fastai.vision import *
from torchvision.ops import box_iou
from .utils import *

__all__ = ['format_bb', 'unformat_bb', 'cdice',
           'Heatmap', 'Heatmaps', 'HeatmapLbl', 'HeatmapsLbl', 'Mask', 'Masks', 'BoundingBox', 'BoundingBoxes', 'BoundingBoxLbl', 'BoundingBoxesLbl']

def format_bb(bb:Tensor, size:int)->Tensor:
    '''Go from x1,y1,x2,y2 to top,left,bottom,right, scaled between -1 and 1.
       Format expected for fastai bounding box item.'''
    return bb[...,[1,0,3,2]].div(size).mul(2).sub(1)

def unformat_bb(bb:Tensor, size:int)->Tensor:
    'Reverse `format_bb`.'
    return bb[...,[1,0,3,2]].add(1).div(2).mul(size)

def draw_bb(bb:Tensor, ax:plt.Axes, color:str='red', lw:int=2, edges:bool=True, fill:bool=False, alpha:float=0.5)->plt.Axes:
    bb = np.array([bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1]])
    if edges:
        ax.add_patch(patches.Rectangle(bb[:2], *bb[-2:], fill=False, edgecolor='black', lw=lw+1))
        ax.add_patch(patches.Rectangle(bb[:2], *bb[-2:], fill=False, edgecolor=color, lw=lw))

    if fill:
        ax.add_patch(patches.Rectangle(bb[:2], *bb[-2:], fill=True, fc=color, alpha=alpha))

    return ax

def mask2bb(x:Tensor)->Tensor:
    x = tensor(x)
    w = torch.where(x.sum(0)>0)[0]
    h = torch.where(x.sum(1)>0)[0]
    if len(w) > 0: x1,x2 = w.min(),w.max()
    else         : x1,x2 = tensor(0,0)
    if len(h) > 0: y1,y2 = h.min(),h.max()
    else         : y1,y2 = tensor(0,0)
    return tensor(x1,y1,x2,y2)

def _contour2bb(x:np.ndarray)->Tensor:
    x = LongTensor(x).squeeze(1)
    (x1,y1),(x2,y2) = x.min(0)[0],x.max(0)[0]
    return tensor(x1,y1,x2,y2)

def _resize_bb(x:Tensor, init_size:Ints, target_size:Ints, inplace:bool=False)->Tensor:
    if not inplace: x = x.clone()
    init_size = listify(init_size, 2)
    target_size = listify(target_size, 2)
    if init_size != target_size:
        fx = target_size[1] / init_size[1]
        fy = target_size[0] / init_size[0]
        x[[0,2]] = x[[0,2]].mul(fx)
        x[[1,3]] = x[[1,3]].mul(fy)

    return x

def cdice(input:Tensor, targs:Tensor)->Rank0Tensor:
    union = input.mul(targs).mul(2)
    inter = input.pow(2)+targs.pow(2)
    return union.sum() / inter.sum()

def build_heatmap(anchors:Tensor, scores:Tensor, size:int, final_size:Optional[int]=None, average:bool=False)->Tensor:
    '''
    Inputs:
        - anchors: [n_anchors, n_classes, 4]
        - scores : [n_anchors, n_classes]
    Output:
        - heatmap: [n_classes, sz, sz]
    '''
    device = anchors.device
    n_anchors,n_classes,_ = anchors.shape
    anchors = anchors.round().long()
    heatmap = torch.zeros(n_classes,size,size).float().to(device)
    for i in range(n_classes):
        for anchor,score in zip(anchors[:,i],scores[:,i]):
            x1,y1,x2,y2 = anchor
            heatmap[i,y1:y2,x1:x2].add_(score)

    if average: heatmap.div_(n_anchors)
    if (final_size is not None) and (size != final_size):
        heatmap = F.interpolate(heatmap[None], size=(final_size,final_size), mode='bilinear', align_corners=False)[0]

    return heatmap

class ImageBase():
    def __init__(self, x:Tensor): self.x = x
    @property
    def shape(self)->torch.Size: return self.x.shape
    @property
    def dims(self)->int: return self.x.dim()
    def __repr__(self)->str: return f'{self.__class__.__name__}({list(self.shape)})'
    def _repr_png_(self): return self._repr_image_format('png')
    def _repr_jpeg_(self): return self._repr_image_format('jpeg')

    def _repr_image_format(self, format_str):
        with BytesIO() as str_buffer:
            plt.imsave(str_buffer, self.x.cpu(), format=format_str, cmap='magma')
            return str_buffer.getvalue()

    def get_resized_x(self, sz:Ints)->Tensor:
        sz = torch.Size(listify(sz, 2))
        x = self.x.float()
        if sz!=self.shape:
            assert self.dims<4, 'Too many dims.'
            add_dims = 4-self.dims
            for i in range(add_dims): x = x[None]
            x = F.interpolate(x, size=sz, mode='bilinear', align_corners=False)
            x = x.squeeze(dim=0).squeeze(dim=0)

        return x

    def resize(self, sz:Ints):
        self.x = self.get_resized_x(sz)
        return self

    def cdice(self, other:Union['ImageBase','ImageListBase','BoundingBox','BoundingBoxes'])->Tensor:
        if isinstance(other, BoundingBox): other = other.to_mask()
        if isinstance(other, BoundingBoxes): other = other.to_masks()
        multi_target = hasattr(other, 'objects')
        sz = other[0].shape[0] if multi_target else other.shape[0]
        a = self.get_resized_x(sz).float()
        b = other.get_resized_x(sz).float()
        if multi_target: out = tensor([cdice(a,o) for o in b])
        else           : out = cdice(a,b)

        return out

    @delegates(plt.Axes.imshow)
    def show(self, ax:Optional[plt.Axes]=None, cmap:str='magma', figsize=(5,5), hide_axis:bool=True, sz:Optional[Ints]=None,
             title:Optional[str]=None, **kwargs:Any)->plt.Axes:
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        x = self.x if sz is None else self.get_resized_x(sz)
        ax.imshow(x.cpu(), cmap=cmap, **kwargs)
        if hide_axis: ax.set_axis_off()
        if title is not None: ax.set_title(title)
        return ax

class ListBase():
    def __init__(self, objects:Collection[Any]):
        self.objects = objects

    def __repr__(self)->str:
        out = ', '.join([str(o) for o in self.objects[:5]])
        n = len(self)
        if n > 5: out += ', ...'
        return f'{self.__class__.__name__}({out}) [{n} elements]'

    def __getitem__(self,i:int)->Any: return self.objects[i]
    def __len__(self)->int: return len(self.objects)
    def __iter__(self): return iter(self.objects)

    def to_tensor(self, dim:int=0)->Tensor:
        return torch.stack([o.x for o in self], dim=dim)

class ImageListBase(ListBase):
    def get_resized_x(self, sz:Ints)->Tensor:
        sz = torch.Size(listify(sz, 2))
        same_sizes = len({o.shape for o in self})==1
        if same_sizes:
            x = self.to_tensor().float()
            if self[0].shape != sz:
                dims = x.dim()
                if dims < 4: x = x[:,None]
                x = F.interpolate(x, size=sz, mode='bilinear', align_corners=False)
                if dims < 4: x = x[:,0]

        else:
            x = torch.stack([o.get_resized_x(sz) for o in self], dim=0)

        return x

    def resize(self, sz:Ints):
        sz = torch.Size(listify(sz, 2))
        same_sizes = len({o.shape for o in self})==1
        if same_sizes:
            if self[0].shape != sz:
                x = self.to_tensor()
                dims = x.dim()
                if dims < 4: x = x[:,None]
                x = F.interpolate(x, size=sz, mode='bilinear', align_corners=False)
                if dims < 4: x = x[:,0]
                for i,o in enumerate(x): self[i].x = o

        else:
            for o in self: o.resize(sz)

        return self

    def cdice(self, other:Union['ImageBase','ImageListBase','BoundingBox','BoundingBoxes'])->Tensor:
        if isinstance(other, BoundingBox): other = other.to_mask()
        if isinstance(other, BoundingBoxes): other = other.to_masks()
        multi_target = hasattr(other, 'objects')
        sz = other[0].shape[0] if multi_target else other.shape[0]
        a = self.get_resized_x(sz).float()
        b = other.get_resized_x(sz).float()
        if multi_target: out = [cdice(o,oo) for o,oo in zip(a,b)]
        else           : out = [cdice(o,b) for o in a]
        return tensor(out)

    @delegates(ImageBase.show)
    def show(self, cols:int=5, size:int=3, cmap:str='magma', axs:Optional[plt.Axes]=None, hide_axis:bool=True,
             **kwargs:Any)->Collection[plt.Axes]:
        cols = min(len(self), cols)
        n = math.ceil(len(self) / cols)
        if axs is None: fig,axs = plt.subplots(n, cols, figsize=(size*cols,size*n))
        for ax,o in zip(axs.flatten(),self): o.show(ax=ax, cmap=cmap, hide_axis=hide_axis, **kwargs)
        if hide_axis:
            for ax in axs.flatten(): ax.axis('off')

        return axs

class Heatmap(ImageBase):
    def __init__(self, x:Tensor):
        assert x.dim() == 2, f'Invalid number of dimensions: {x.dim()}'
        super().__init__(x)
        self._orig_scale = None

    @classmethod
    def from_preds(cls, anchors:Tensor, scores:Tensor, size:int, final_size:Optional[int]=None, average:bool=False):
        assert anchors.size(1) == 1, 'For more than one class use `HeatmapLbl`'
        return cls(build_heatmap(anchors=anchors, scores=scores, size=size, final_size=final_size, average=average)[0])

    def clone(self): return self.__class__(self.x.clone())

    @property
    def value_range(self)->Tensor: return tensor(self.x.min(),self.x.max())

    def scale(self, vmin:Optional[float]=None, vmax:Optional[float]=None):
        'if no attributes are given it will scale the heatmap between 0 and 1.'
        vmin = ifnone(vmin, self.x.min())
        vmax = ifnone(vmax, self.x.max())
        diff = vmax - vmin
        self.x = (self.x + vmin) / diff
        self._orig_scale = (vmin,diff)
        return self

    def unscale(self):
        assert self._orig_scale is not None, 'Scale hasnt been applied.'
        vmin,diff = self._orig_scale
        self.x = (self.x * diff) - vmin
        self._orig_scale = None
        return self

    def get_mask(self, th:float=0.75, percentile:bool=True)->'Mask':
        x = self.x
        if percentile: th = get_percentile(x, th)
        mask = x>=th if th!=x.min() else x>th
        return Mask(mask.byte())

    def to_bounding_box(self, th:float=0.75, percentile:bool=True)->'BoundingBox':
        return self.get_mask(th=th, percentile=percentile).to_bounding_box()

    def get_contours(self, th:float=0.75, percentile:bool=True)->'BoundingBoxes':
        return self.get_mask(th=th, percentile=percentile).get_contours()

    def get_most_active_contour(self, th:float=0.75, percentile:bool=True)->'BoundingBox':
        return self.get_mask(th=th, percentile=percentile).get_most_active_contour(self)

    @delegates(sns.heatmap)
    def show_heatmap(self, sz:int=16, ax:Optional[plt.Axes]=None, cmap:str='magma', figsize=(6,6), hide_axis:bool=True, annot:bool=True,
                     cbar:bool=False, **kwargs:Any)->plt.Axes:
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        x = self.get_resized_x(sz).cpu()
        sns.heatmap(x, cmap=cmap, annot=annot, ax=ax, cbar=cbar, **kwargs)
        if hide_axis: ax.set_axis_off()
        ax.set_ylim(ax.get_ylim()[0]+0.5,0)
        return ax

class Heatmaps(ImageListBase):
    def __init__(self, objects:Collection[Heatmap]):
        self.objects = objects

    @classmethod
    def from_tensor(cls, heatmaps:Tensor, **kwargs:Any):
        'Format = [bs,...,sz,sz]'
        heatmaps = heatmaps.squeeze()
        if heatmaps.dim() == 2: heatmaps = heatmaps[None]
        assert heatmaps.dim() == 3, f'Invalid number of dimensions: {heatmaps.dim()}.'
        return cls([Heatmap(o) for o in heatmaps], **kwargs)

    @classmethod
    def from_preds(cls, anchors:Collection[Tensor], scores:Collection[Tensor], size:int, final_size:Optional[int]=None, average:bool=False,
                   **kwargs:Any):
        assert all(anchor.size(1) == 1 for anchor in anchors), 'For more than one class use `HeatmapsLbl`'
        return cls([Heatmap.from_preds(anchors=anchor, scores=score, size=size, final_size=final_size, average=average)
                    for anchor,score in zip(anchors,scores)], **kwargs)

    def clone(self): return self.__class__([o.clone() for o in self])

    @property
    def value_range(self)->Tensor: return torch.stack([o.value_range for o in self])

    def scale(self, vmins:Optional[Floats]=None, vmaxs:Optional[Floats]=None):
        vmins = listify(ifnone(vmins, [None]), len(self))
        vmaxs = listify(ifnone(vmaxs, [None]), len(self))
        for o,vmin,vmax in zip(self,vmins,vmaxs): o.scale(vmin, vmax)
        return self

    def unscale(self):
        for o in self: o.unscale()
        return self

    def get_masks(self, th:float=0.75, percentile:bool=True)->'Masks':
        return Masks([o.get_mask(th=th, percentile=percentile) for o in self])

    def to_bounding_boxes(self, th:float=0.75, percentile:bool=True)->'BoundingBoxes':
        return BoundingBoxes([o.to_bounding_box(th=th, percentile=percentile) for o in self])

    def get_contours(self, th:Union[float,Floats]=0.75, percentile:bool=True)->Collection['BoundingBoxes']:
        return [o.get_contours(th=t, percentile=percentile) for o,t in zip(self,listify(th,len(self)))]

    def get_most_active_contours(self, th:Union[float,Floats]=0.75, percentile:bool=True)->'BoundingBoxes':
        return BoundingBoxes([o.get_most_active_contour(th=t, percentile=percentile) for o,t in zip(self,listify(th,len(self)))])

class HeatmapLbl(Heatmaps):
    def __init__(self, objects:Collection[Heatmap], c:Optional[int]=None, classes:Optional[Collection[str]]=None):
        super().__init__(objects)
        if (c is None) and (classes is None): c = len(objects)
        else                                : c = ifnone(c, len(classes))
        classes = ifnone(classes, list(range(c)))
        self.c,self.classes = c,classes
        self.lbl2idx = {lbl:i for i,lbl in enumerate(self.classes)}

    def __repr__(self)->str: return super().__repr__() + f'\n{self.c} classes ({self.classes})'
    def select_lbl(self, label:str)->Heatmap: return self[self.lbl2idx[label]]

    def clone(self): return self.__class__([o.clone() for o in self], self.c, self.classes)

    def get_class(self, c:str)->Optional[Heatmap]:
        for i,o in enumerate(self.classes):
            if o==c: return self[i]

    def filter_classes(self, classes:Collection[str])->'HeatmapLbl':
        res = [o for o,lbl in zip(self,self.classes) if lbl in classes]
        return self.__class__(res, classes=classes)

    def get_value_range(self, labeled:bool=True)->Tensor:
        out = self.value_range
        if labeled: out = {lbl:o for o,lbl in zip(out,self.classes)}
        return out

    def scale_from_dict(self, d:dict):
        'Expected format {lbl: {min: vmin, max: vmax}, ...}'
        for k in self.classes: self.select_lbl(k).scale(*d[k])
        return self

    def get_contours(self, th:Union[float,dict,Floats]=0.75, percentile:bool=True, default_th:float=0.75)->Collection['BoundingBoxes']:
        if isinstance(th, dict): th = [th.get(c, default_th) for c in self.classes]
        return super().get_contours(th, percentile)

    def get_most_active_contours(self, th:Union[float,dict,Floats]=0.75, percentile:bool=True, default_th:float=0.75)->'BoundingBoxLbl':
        if isinstance(th, dict): th = [th.get(c, default_th) for c in self.classes]
        return BoundingBoxLbl([o.get_most_active_contour(th=t, percentile=percentile) for o,t in zip(self,listify(th,len(self)))], self.classes)

    @delegates(Heatmaps.show)
    def show(self, cols:int=5, size:int=3, cmap:str='magma', axs:Optional[plt.Axes]=None, hide_axis:bool=True, labels=True,
             **kwargs:Any)->plt.Axes:
        axs = super().show(cols=cols, size=size, cmap=cmap, axs=axs, hide_axis=hide_axis, **kwargs)
        for ax,t in zip(axs.flatten(), self.classes): ax.set_title(t)
        return axs

class HeatmapsLbl(ListBase):
    def __init__(self, objects:Collection[HeatmapLbl]):
        classes = objects[0].classes
        assert all(classes == o.classes for o in objects), 'not all classes are equal'
        super().__init__(objects)
        self.classes = classes

    @property
    def value_range(self)->Tensor:
        x = torch.stack([o.value_range for o in self])
        return torch.stack([x[...,0].min(0)[0],x[...,1].max(0)[0]], dim=-1)

    def get_value_range(self, labeled:bool=True)->Tensor:
        out = self.value_range
        if labeled: out = {lbl:o for o,lbl in zip(out,self.classes)}
        return out

    def scale(self, vmins:Optional[Floats]=None, vmaxs:Optional[Floats]=None):
        for o in self: o.scale(vmins, vmaxs)
        return self

    def scale_from_dict(self, d:dict):
        'Expected format {lbl: {min: vmin, max: vmax}, ...}'
        for o in self: o.scale_from_dict(d)
        return self

    def unscale(self):
        for o in self: o.unscale()
        return self

    @classmethod
    def from_tensor(cls, heatmaps:Tensor, classes:Optional[Collection[str]]=None, **kwargs:Any):
        'Format = [bs,c,sz,sz]'
        assert heatmaps.dim() == 4, f'Invalid number of dimensions: {heatmaps.dim()}.'
        return cls([HeatmapLbl.from_tensor(o, classes=classes, **kwargs) for o in heatmaps])

    @classmethod
    def from_preds(cls, anchors:Collection[Collection[Tensor]], scores:Collection[Collection[Tensor]], size:int, final_size:Optional[int]=None,
                   average:bool=False, classes:Optional[Collection[str]]=None):
        bs = len(anchors[0])
        assert all(len(o)==bs for o in anchors)
        res = [HeatmapLbl.from_preds([o[i] for o in anchors], [o[i] for o in scores], size, classes=classes) for i in range(bs)]
        return cls(res)

    def get_most_active_contours(self, th:Union[float,dict,Floats]=0.75, percentile:bool=True, default_th:float=0.75)->'BoundingBoxesLbl':
        if isinstance(th, dict): th = [th.get(c, default_th) for c in self.classes]
        res = [o.get_most_active_contours(th, percentile=percentile) for o in self]
        return BoundingBoxesLbl(res)

    def to_tensor(self, dim:int=0)->Tensor:
        return torch.stack([o.to_tensor(dim=dim) for o in self], dim=dim)

class Mask(ImageBase):
    def __init__(self, x:Tensor):
        assert x.dtype == torch.uint8, f'Invalid format: {x.dtype}'
        super().__init__(x)

    @classmethod
    def from_heatmap(cls, x:Union[Tensor,Heatmap], th:float=0.75, percentile:bool=True):
        if not isinstance(x, Heatmap): x = Heatmap(x)
        return x.get_mask(th=th, percentile=percentile)

    def is_empty(self)->bool: return (self.x.sum() == 0).item()
    def to_bounding_box(self)->'BoundingBox': return BoundingBox(mask2bb(self.x), sz=self.shape)

    @delegates(ImageBase.show)
    def show(self, ax:Optional[plt.Axes]=None, cmap:str='magma', figsize=(5,5), hide_axis:bool=True, contours:bool=False,
             **kwargs:Any)->plt.Axes:
        ax = super().show(ax=ax, cmap=cmap, figsize=figsize, hide_axis=hide_axis, **kwargs)
        if contours: self.get_contours().draw(ax)
        return ax

    def get_contours(self, min_area:int=0)->'BoundingBoxes':
        if self.is_empty(): return BoundingBoxes([self.to_bounding_box()])
        # contours,_ = cv2.findContours(self.x.cpu().numpy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours,_ = cv2.findContours(self.x.cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==1: return BoundingBoxes([self.to_bounding_box()])
        return BoundingBoxes([oo for oo in [BoundingBox(_contour2bb(o), sz=self.shape) for o in contours] if oo.area >= min_area])
    
    def get_most_active_contour(self, hm:Heatmap)->'BoundingBox':
        return self.get_contours().get_most_active(hm)

class Masks(ImageListBase):
    def __init__(self, objects:Collection[Mask]):
        super().__init__(objects)

    @classmethod
    def from_heatmaps(cls, x:Union[Tensor,Heatmaps], th:float=0.75, percentile:bool=True):
        if not isinstance(x, Heatmaps): x = Heatmaps(x)
        return x.get_masks(th=th, percentile=percentile)

    def to_bounding_boxes(self)->'BoundingBoxes':
        return BoundingBoxes([o.to_bounding_box() for o in self])

    def get_contours(self)->Collection['BoundingBoxes']:
        return [o.get_contours() for o in self]
    
    def get_most_active_contours(self, hms:Heatmaps)->'BoundingBoxes':
        assert len(self) == len(hms)
        return BoundingBoxes([o.get_most_active_contours(hm) for o,hm in zip(self,hms)])

class BoundingBox():
    def __init__(self, x:Tensor, sz:Ints):
        'Format = [x1,y1,x2,y2]'
        assert x.dim() == 1, f'Invalid number of dimensions: {x.dim()}.'
        assert len(x) == 4, f'Invalid format.'
        sz = listify(sz, 2)
        self.x,self.sz = x.float(),sz
    
    def __repr__(self)->str:
        bb = ', '.join([f'{o:.2f}' for o in self.x])
        return f'{self.__class__.__name__}(({bb}), area={self.area:.2f}, sz={self.sz})'

    @classmethod
    def from_mask(cls, mask:Union[Tensor,Mask]):
        if not isinstance(mask, Mask): mask = Mask(mask)
        return mask.to_bounding_box()

    @classmethod
    def from_preds(cls, x:Tensor, sz:int):
        'Format = [bs,(y1,x1,y2,x2)]'
        return cls(unformat_bb(x, sz)[0], sz)

    @property
    def x1(self)->Rank0Tensor: return self.x[0]
    @property
    def x2(self)->Rank0Tensor: return self.x[2]
    @property
    def y1(self)->Rank0Tensor: return self.x[1]
    @property
    def y2(self)->Rank0Tensor: return self.x[3]
    @property
    def area(self)->Rank0Tensor: return (self.x2-self.x1) * (self.y2-self.y1)
    @property
    def shape(self)->torch.Size: return torch.Size(self.sz)

    def to_mask(self, sz:Optional[Ints]=None)->Mask:
        sz = listify(ifnone(sz, self.sz), 2)
        x1,y1,x2,y2 = self.get_resized_x(sz).long()
        x = torch.zeros(*sz, dtype=torch.uint8).to(x1.device)
        x[y1:y2, x1:x2] = 1
        return Mask(x)

    def get_unformated_data(self)->Tensor:
        'Format as in the model output (y1,x1,y2,x2) between 0 and 1.'
        return format_bb(self.x[None], self.sz[0])[0]

    def get_resized_x(self, sz:Ints)->Tensor: return _resize_bb(self.x, init_size=self.sz, target_size=sz)

    def resize(self, sz:Ints):
        _resize_bb(self.x, init_size=self.sz, target_size=sz, inplace=True)
        self.sz = listify(sz, 2)
        return self

    def draw(self, ax:plt.Axes, color:str='red', lw:int=2, edges:bool=True, fill:bool=False, alpha:float=0.5, resize:Optional[int]=None)->plt.Axes:
        x = self.x if resize is None else self.get_resized_x(resize)
        return draw_bb(x, ax=ax, color=color, lw=lw, edges=edges, fill=fill, alpha=alpha)

    def get_max_value(self, hm:Heatmap)->Rank0Tensor:
        'Get the max value of the `BoundingBox` according to a `Heatmap`.'
        x1,y1,x2,y2 = self.get_resized_x(hm.shape).long()
        values = hm.x[y1:y2, x1:x2]
        return values.max() if values.numel()>0 else hm.x.min()

    def intersection(self, other:'BoundingBox')->Rank0Tensor:
        a = self.x
        b = other.get_resized_x(self.sz)
        lt = torch.max(a[:2], b[:2])
        rb = torch.min(a[2:], b[2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[0]*wh[1]
        return inter 

    def iou(self, other:Union['BoundingBox','BoundingBoxes'])->Rank0Tensor:
        if isinstance(other, BoundingBoxes): return other.iou(self)
        a = self.x[None]
        b = other.get_resized_x(self.sz)[None].to(a.device)
        return box_iou(a,b).item()

    def iobb(self, other:'BoundingBox')->Rank0Tensor:
        'bb area is given by `other`'
        return self.intersection(other) / other.area

class BoundingBoxes(ListBase):
    def __init__(self, objects:Collection[BoundingBox]):
        self.objects = objects
        self._all_sz_eq = self._check_szs()

    @classmethod
    def from_tensor(cls, boxes:Tensor, sz:Ints, **kwargs:Any):
        'Format = [bs,(x1,y1,x2,y2)]'
        boxes = boxes.squeeze()
        if boxes.dim() == 1: boxes = boxes[None]
        assert boxes.dim() == 2, f'Invalid number of dimensions: {boxes.dim()}.'
        assert boxes.size(-1) == 4, f'Invalid format.'
        sz = listify(sz, 2)
        return cls([BoundingBox(o,sz) for o in boxes], **kwargs)

    @classmethod
    def from_preds(cls, x:Tensor, sz:int):
        'Format = [bs,(y1,x1,y2,x2)]'
        return cls.from_tensor(unformat_bb(x, sz), sz)

    def _check_szs(self):
        sz = self[0].sz
        return all(o.sz==sz for o in self)

    @property
    def x1(self)->Tensor:
        assert self._all_sz_eq, 'Not all bounding boxes have the same size.'
        return tensor([o.x1 for o in self])

    @property
    def x2(self)->Tensor:
        assert self._all_sz_eq, 'Not all bounding boxes have the same size.'
        return tensor([o.x2 for o in self])

    @property
    def y1(self)->Tensor:
        assert self._all_sz_eq, 'Not all bounding boxes have the same size.'
        return tensor([o.y1 for o in self])

    @property
    def y2(self)->Tensor:
        assert self._all_sz_eq, 'Not all bounding boxes have the same size.'
        return tensor([o.y2 for o in self])

    @property
    def area(self)->Tensor:
        assert self._all_sz_eq, 'Not all bounding boxes have the same size.'
        return (self.x2-self.x1) * (self.y2-self.y1)

    def draw(self, ax:plt.Axes, color:Optional[str]=None, cmap:str='tab10', **kwargs:Any)->plt.Axes:
        cm = plt.cm.cmap_d[cmap]
        for i,o in enumerate(self): o.draw(ax, color=ifnone(color, cm(i)), **kwargs)
        return ax

    def to_masks(self, sz:Optional[Ints]=None)->Masks:
        sz = ifnone(sz, self[0].sz)
        return Masks([o.to_mask(sz) for o in self])

    def to_tensor(self, sz:Optional[Ints]=None)->Tensor:
        sz = ifnone(sz, self[0].sz)
        data = [o.get_resized_x(sz) for o in self]
        return torch.stack(data, dim=0)

    def get_unformated_data(self)->Tensor:
        'Format as in the model output.'
        return format_bb(self.to_tensor(), self[0].sz[0])

    def resize(self, sz:Ints):
        for o in self: o.resize(sz)
        return self

    def get_max_values(self, hm:Heatmap)->Rank0Tensor:
        'Get the max value of the `BoundingBox` according to a `Heatmap`.'
        return torch.stack([o.get_max_value(hm) for o in self])

    def get_most_active(self, hm:Heatmap)->BoundingBox:
        'Get the most active `BoundingBox` according to a `Heatmap`.'
        max_values = self.get_max_values(hm)
        return self[max_values.argmax()]

    def _iou_box(self, other:'BoundingBox')->Tensor:
        a = self.to_tensor(other.sz)
        b = other.x[None].to(a.device)
        return box_iou(a,b).squeeze(-1)

    def _iou_boxes(self, other:'BoundingBoxes')->Tensor:
        sz = other[0].sz
        assert len(self) == len(other)
        a = self.to_tensor(sz).cpu().unsqueeze(1)
        b = other.to_tensor(sz).cpu().unsqueeze(1)
        return torch.cat([box_iou(i,j) for i,j in zip(a,b)]).squeeze()

    def iou(self, other:Union['BoundingBox','BoundingBoxes'])->Tensor:
        if isinstance(other, BoundingBox): return self._iou_box(other)
        if isinstance(other, BoundingBoxes): return self._iou_boxes(other)
        else: raise Exception('Invalid format.')

    def iobb(self, other:'BoundingBoxes')->Tensor:
        'bb area is given by `other`'
        assert len(self) == len(other)
        return tensor([a.iobb(b) for a,b in zip(self,other)])

    def multi_iou(self, other:'BoundingBox')->Rank0Tensor:
        other = other.resize(self[0].sz)
        inter = sum([o.intersection(other) for o in self])
        union = sum([o.area for o in self]) + other.area
        return inter / ( union - inter)

class BoundingBoxLbl(BoundingBoxes):
    def __init__(self, objects:Collection[BoundingBox], classes:Optional[Collection[str]]=None):
        super().__init__(objects)
        self.classes = ifnone(classes, range_of(objects))

    def __repr__(self)->str: return super().__repr__() + f'\nclasses ({self.classes})'

    @classmethod
    def from_preds(cls, pred:Tuple[Tensor,Tensor], sz:int, classes:Optional[Collection[str]]=None):
        boxes,labels = pred
        boxes,labels = boxes.cpu(),tensor(labels).cpu()
        tmp = tensor([[-1.,-1.,1.,1.]])
        idxs = []
        for i,(box,lbl) in enumerate(zip(boxes,labels)):
            if ((box==tmp).sum() != 4) and (lbl > 0): idxs.append(i)

        idxs = LongTensor(idxs)
        res = cls.from_tensor(unformat_bb(boxes[idxs], sz), sz, classes=classes)
        c = listify(labels[idxs])
        res.classes = c if classes is None else [classes[o] for o in c]
        return res

    def _iou_box_lbl(self, other:'BoundingBoxLbl')->Tensor:
        sz = other[0].sz
        return tensor([self.get_class(c).iou(other.get_class(c)) for c in self.classes])

    def iou(self, other:Union['BoundingBox', 'BoundingBoxLbl','BoundingBoxes'])->Tensor:
        if isinstance(other, BoundingBoxLbl): return self._iou_box_lbl(other)
        super().iou(other)

    def get_class(self, c:str)->Optional[BoundingBox]:
        for i,o in enumerate(self.classes):
            if o==c: return self[i]

    def filter_classes(self, classes:Collection[str])->'BoundingBoxLbl':
        res = [o for o,lbl in zip(self,self.classes) if lbl in classes]
        return self.__class__(res, classes)

class BoundingBoxesLbl(ListBase):
    def __init__(self, objects:Collection[BoundingBoxLbl]):
        super().__init__(objects)

    @classmethod
    def from_tensor(cls, x:Tensor, sz:int, classes:Optional[Collection[str]]=None):
        'Format = [bs,c,4]'
        res = [BoundingBoxLbl.from_tensor(o, sz, classes=classes) for o in x]
        return cls(res)

    @classmethod
    def from_gumbel_hook(cls, x:Collection[Tensor], sz:int, classes:Optional[Collection[str]]=None):
        'Format = [bs,1,1,4]*c'
        res = torch.stack(x, dim=1).squeeze(2).squeeze(2)
        return cls([BoundingBoxLbl.from_tensor(o, sz, classes=classes) for o in res])

    @classmethod
    def from_preds(cls, preds:Tuple[Tensor,Tensor], sz:int, classes:Optional[Collection[str]]=None):
        res = [BoundingBoxLbl.from_preds(pred, sz, classes) for pred in zip(*preds)]
        return cls(res)

    def iou(self, other:'BoundingBoxesLbl')->Collection[Tensor]:
        return [a.iou(b) for a,b in zip(self,other)]

    def get_class(self, c:str)->BoundingBoxes: return BoundingBoxes([o.get_class(c) for o in self])
