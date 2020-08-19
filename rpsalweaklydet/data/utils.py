from fastai.vision import *

__all__ = ['empty_box', 'get_bbx_samples']

def empty_box()->Tensor:
    'The dataset have bounding boxes for only a subset of the data, so we consider boxes that covers all the image as empty.'
    return tensor([[-1.,-1.,1.,1.]])

def get_bbx_samples(dl:DeviceDataLoader, n_samples:int=8, pbar:bool=True, shuffle_dl:bool=True)->Collection[Tensors]:
    'Get samples with bounding boxes from a `DataLoader`.'
    samples_x,samples_y = [],[]
    it = dl.new(shuffle=shuffle_dl)
    if pbar: it = progress_bar(it)
    empty = empty_box().to(dl.device)

    for xb,yb in it:
        idxs = []
        for i,(ybox,ylbl) in enumerate(zip(*yb)):
            if ((ybox[ylbl>0]==empty).sum(1) != 4).any(): idxs.append(i)

        for idx in idxs:
            samples_x.append(xb[None,idx].cpu())
            samples_y.append([yb[0][None,idx].cpu(),yb[1][None,idx].cpu()])
            if len(samples_x)>=n_samples:
                # The final result
                samples_x = torch.cat(samples_x)
                max_boxes = max([o[1].size(1) for o in samples_y])
                boxes  = torch.cat([F.pad(o[0], (0,0,max_boxes-o[1].size(1),0)) for o in samples_y], dim=0)
                labels = torch.cat([F.pad(o[1], (max_boxes-o[1].size(1),0)) for o in samples_y], dim=0)
                samples_y = boxes,labels
                return samples_x,samples_y

    raise Exception(f'Insufficient samples: wanted {n_samples} and got {len(samples_x)}.')
