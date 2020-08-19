from fastai.vision import *

class CustomObjectCategoryProcessor(MultiCategoryProcessor):
    "`PreProcessor` for labelled bounding boxes."
    def __init__(self, ds:ItemList, pad_idx:int=0):
        super().__init__(ds)
        self.pad_idx = pad_idx
        self.state_attrs.append('pad_idx')

    def process(self, ds:ItemList):
        ds.pad_idx = self.pad_idx
        super().process(ds)

    def process_one(self,item): return [item[0], [self.c2i.get(o,None) for o in item[1]]]

    def generate_classes(self, items):
        "Generate classes from unique `items` with no background class."
        classes = super().generate_classes([o[1] for o in items])
        return list(classes)

class CustomObjectCategoryList(ObjectCategoryList):
    _processor = CustomObjectCategoryProcessor
    def analyze_pred(self, pred): return pred[0],pred[1][None].argmax(dim=1)

class CustomObjectItemList(ObjectItemList):
    _label_cls,_square_show_res = CustomObjectCategoryList,False

def multiclass_bb_pad_collate(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    if isinstance(samples[0][1], int): return data_collate(samples)
    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls = s[1].data
        if not (bbs.nelement() == 0):
            bboxes[i,-len(lbls):] = bbs
            labels[i,-len(lbls):] = tensor(lbls)
    return torch.cat(imgs,0), (bboxes,labels)