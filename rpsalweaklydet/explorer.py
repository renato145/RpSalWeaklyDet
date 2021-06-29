from fastai.vision import *
from fastai.callbacks import *
from .structures import *
from .hooks import multi_hook_output,multi_hook_outputs

__all__ = ['SaliencyExplorer']

class SaliencyExplorer:
    def __init__(self, learn:Learner, data:Optional[DataBunch]=None, th_dict:Optional[dict]=None, filename:Optional[str]=None, overwrite:bool=False,
                 img_size:int=512):
        self.learn = learn
        self.data = ifnone(data, learn.data)
        self.th_dict = ifnone(th_dict, 0.9)
        self.threadholds = [o/100 for o in range(60,101)]
        self.img_size = img_size
        self.process(filename, overwrite)
        self._ious = None
        self._cdice = None

    def __repr__(self)->str: return f'{self.__class__.__name__}()'

    def process(self, filename:Optional[str]=None, overwrite:bool=False)->None:
        path = None if filename is None else self.learn.path / f'{filename}.pth'

        if (path is None) or (not path.exists()) or overwrite:
            size = self.img_size
            m = self.learn.model
            m.eval()
            true_boxes,pred_boxes,saliency,detection = [],[],[],[]

            for xb,yb in progress_bar(self.data.valid_dl):
                # Get Prediction
                    # (hook_outputs(m.head.scores_layer) if m.det_head_multi
                    # else multi_hook_output(m.head.scores_layer)) as scores_hook,\
                with torch.no_grad(), hook_outputs([m.saliency_layer]) as hooks,\
                    multi_hook_output(m.apply_deltas) as mhook,\
                    multi_hook_output(m.head.scores_layer) as scores_hook,\
                    hook_output(m.classifier) as crop_logits_hook:
                    yb_ = m(xb.cuda())

                out_saliency, = hooks.stored
                anchors = mhook.stored
                scores = scores_hook.stored

                # Obtain objects
                t_boxes = BoundingBoxesLbl.from_preds(yb, size, self.data.classes)
                p_saliency = HeatmapsLbl.from_tensor(out_saliency.sigmoid().cpu(), classes=self.data.classes[1:])
                hm = HeatmapsLbl.from_preds(anchors, scores, size, classes=self.data.classes[1:])
                p_boxes = hm.get_most_active_contours(self.th_dict)

                true_boxes.append(t_boxes.objects)
                saliency.append([b.filter_classes(a.classes) for a,b in zip(t_boxes,p_saliency)])
                detection.append([b.filter_classes(a.classes) for a,b in zip(t_boxes,hm)])
                pred_boxes.append([b.filter_classes(a.classes) for a,b in zip(t_boxes,p_boxes)])

            d = {'true_boxes': sum(true_boxes, []),
                 'pred_boxes': sum(pred_boxes, []),
                 'saliency'  : sum(saliency  , []),
                 'detection' : sum(detection , [])}
            d['labels'] = [o.classes for o in d['true_boxes']]
            mix = []
            for sal,det in zip(d['saliency'], d['detection']):
                mix.append(HeatmapLbl.from_tensor(det.to_tensor().cpu().mul(sal.get_resized_x(det[0].shape).cpu()), classes=det.classes))

            d['mix'] = mix
            if path is not None: torch.save(d, path)
        else:
            d = torch.load(path)

        self.d = d
        self.labels = sorted(list(set(sum(self.d['labels'], []))))
        ious = {}
        for lbl in self.labels:
            t = self.get_label(lbl)
            ious[lbl] = BoundingBoxes(t['true_boxes']).iou(BoundingBoxes(t['pred_boxes']))

        self.ious = ious

    def process_iou(self, filename:Optional[str]=None, overwrite:bool=False)->DataFrame:
        path = None if filename is None else self.learn.path / f'{filename}.csv'

        if (path is None) or (not path.exists()) or overwrite:
            ious = {}
            mb = master_bar(['saliency', 'detection', 'mix'])
            for input_type in mb:
                mb.main_bar.comment = input_type
                ious[input_type] = {}
                for lbl in progress_bar(self.labels, parent=mb):
                    ious[input_type][lbl] = {}
                    data = self.get_label(lbl, input_type)
                    boxes = self.get_label(lbl, 'true_boxes')
                    for method in ['split', 'union']:
                        mb.child.comment = lbl
                        ious[input_type][lbl][method] = self._compute_iou(data, boxes, percentile=input_type!='saliency', method=method)

            out = [[[(input_type,method,lbl,iou.item(),th) for method,(iou,th) in vv.items()] for lbl,vv in v.items()] for input_type,v in ious.items()]
            out = sum(sum(out, []), [])
            out = DataFrame(out, columns=['input_type', 'method', 'lbl', 'iou', 'th'])
            if path is not None: out.to_csv(path, index=False)
        else:
            out = pd.read_csv(path)

        self._ious = out
        # Get best params
        summary = out.groupby(['input_type', 'method'])['iou'].mean()
        self._ious_best = dict(zip(summary.index.names, summary.idxmax()))
        print(f'Updating data using {self._ious_best}...')
        self.update_iou()
        return out

    def update_iou(self, input_type:Optional[str]=None, method:Optional[str]=None)->None:
        input_type = ifnone(input_type, self._ious_best['input_type'])
        method = ifnone(method, self._ious_best['method'])
        th_dict = self.get_ious(input_type, method)['th']
        if method == 'split':
            pred_boxes = [o.get_most_active_contours(th_dict, percentile=input_type!='saliency') for o in self.d[input_type]]
        elif method == 'union':
            pred_boxes = [o.get_contours(th_dict, percentile=input_type!='saliency') for o in self.d[input_type]]

        self.d['pred_boxes'] = pred_boxes

        for lbl in self.labels:
            t = self.get_label(lbl)
            true_boxes = BoundingBoxes(t['true_boxes'])
            pred_boxes = t['pred_boxes']
            if   method == 'split': self.ious[lbl] = true_boxes.iou(BoundingBoxes(pred_boxes))
            elif method == 'union': self.ious[lbl] = tensor([a.multi_iou(b) for a,b in zip(pred_boxes, true_boxes)])

    def _compute_iou(self, data:Heatmaps, boxes:BoundingBoxes, percentile:bool, method:str):
        d = []
        for th in self.threadholds:
            if method == 'split':
                pred_boxes = data.get_most_active_contours(th, percentile=percentile)
                d.append(pred_boxes.iou(boxes).mean())
            elif method == 'union':
                ious = tensor([a.multi_iou(b) for a,b in zip(data.get_contours(th, percentile=percentile), boxes)]).mean()
                d.append(ious)

        d = tensor(d)
        idx = d.argmax()
        return d[idx],self.threadholds[idx]

    def get_ious(self, input_type:Optional[str]=None, method:Optional[str]=None)->Dict[str,dict]:
        if self._ious is None: return {'iou': {k:v.mean().item() for k,v in self.ious.items()}, 'th': self.th_dict}
        input_type = ifnone(input_type, self._ious_best['input_type'])
        method = ifnone(method, self._ious_best['method'])
        return self._ious.query(f'input_type == {input_type!r} and method == {method!r}').set_index('lbl')[['iou','th']].to_dict()

    def process_cdice(self)->DataFrame:
        cdice = {}
        mb = master_bar(['saliency', 'detection', 'mix'])
        for input_type in mb:
            mb.main_bar.comment = input_type
            cdice[input_type] = {}
            for lbl in progress_bar(self.labels, parent=mb):
                data = self.get_label(lbl, input_type)
                boxes = self.get_label(lbl, 'true_boxes')
                if input_type != 'saliency': data.scale()
                # cdice[input_type][lbl] = data.cdice(boxes).mean()
                cdice[input_type][lbl] = boxes.to_masks().cdice(data).mean()

        out = [[(input_type,lbl,o.item()) for lbl,o in v.items()] for input_type,v in cdice.items()]
        out = sum(out, [])
        out = DataFrame(out, columns=['input_type', 'lbl', 'cdice'])
        self._cdice = out
        return out

    def _get_idxs(self, idxs:dict, field:Optional[str]=None)->dict:
        if field is not None: return {field: [self.d[field][i][j] for i,j in idxs.items()]}
        return {k:[v[i][j] for i,j in idxs.items()] for k,v in self.d.items()}

    def get_label(self, lbl:str, field:Optional[str]=None)->dict:
        idxs = [i for i,o in enumerate(self.d['labels']) if lbl in o]
        idxs = {idx:self.d['labels'][idx].index(lbl) for idx in idxs}
        d = self._get_idxs(idxs, field)
        for k,v in d.items():
            if k == 'true_boxes':
                d[k] = BoundingBoxes(v)
            elif k=='pred_boxes':
                if isinstance(v[0], BoundingBox): d[k] = BoundingBoxes(v)
            elif k in ['saliency', 'detection', 'mix']:
                d[k] = Heatmaps(v)

        return d if field is None else d[field]

    def topk_iou(self, k:Optional[int]=None, label:Optional[str]=None, ascending:bool=False):
        out = {}
        for lbl in listify(ifnone(label, self.labels)):
            ious = self.ious[lbl]
            idxs = ious.topk(ifnone(k, ious.size(0)), largest=not ascending)[1]
            d = self.get_label(lbl)
            d.pop('labels')
            for dk,v in d.items(): d[dk] = [v[i] for i in idxs]
            d['ious'] = ious[idxs]
            d['idxs'] = idxs
            out[lbl] = d

        return out

    def show(self, label:str, idx:int=0, ax:Optional[plt.Axes]=None, figsize:Optional[Tuple[int,int]]=(8,8), alpha=0.35)->plt.Axes:
        if ax is None: fig,ax = plt.subplots(figsize=figsize)
        d = self.topk_iou(label=label)[label]
        idxs = [i for i,o in enumerate(self.d['labels']) if label in o]
        x = self.data.valid_ds.x[idxs[idx]]
        y = self.data.valid_ds.inner_df.iloc[idx].label[1]
        with torch.no_grad(): y_ = self.learn.model.eval()(self.data.one_item(x)[0])[1][0,1].sigmoid().cpu()
        pred = y_[self.data.c2i[label]-1]
        sm = d['mix'][idx]
        tb = d['true_boxes'][idx]
        pb = d['pred_boxes'][idx]
        iou = d['ious'][idx]
        # Plot
        x.show(ax)
        sm.show(ax, alpha=alpha)
        pb.draw(ax, color='yellow')
        tb.draw(ax, color='#00ef00')
        ax.set_title(f'{label}\nScore={pred:.4f} - IoU={iou:.2f}', size=16)
        return ax

    def show_top(self, k:int=3, label:Optional[str]=None, ascending:bool=False, hm:bool=False, cols:int=4, size:int=4)->Collection[plt.Axes]:
        if hm: return self.show_top_hm(k=k, label=label, ascending=ascending, size=size)
        cols = min(k, cols)
        labels = listify(ifnone(label, self.labels))
        n = len(labels)
        lbl_rows = math.ceil(k/cols)
        naxs_lbl = lbl_rows * cols
        rows = lbl_rows * n
        fig,axs = plt.subplots(rows, cols, figsize=(size*cols,size*rows))
        axs = axs.flatten()
        for ax in axs: ax.set_axis_off()
        for i,lbl in enumerate(labels):
            data = self.topk_iou(k, lbl, ascending=ascending)[lbl]
            idxs = [i for i,o in enumerate(self.d['labels']) if lbl in o]
            idxs = [idxs[i] for i in data['idxs']]
            lbl_axs = axs[i*naxs_lbl:(i+1)*naxs_lbl]
            for ax,idx,iou,true_box,pred_box in zip(lbl_axs,idxs,data['ious'],data['true_boxes'],data['pred_boxes']):
                self.data.valid_ds.x[idx].show(ax=ax)
                ax.set_title(f'{lbl}\niou={iou:.4f}')
                pred_box.draw(ax, color='green', fill=True)
                true_box.draw(ax, color='white', fill=True)

        return axs

    def show_top_hm(self, k:int=3, label:Optional[str]=None, ascending:bool=False, size:int=4)->Collection[plt.Axes]:
        m = self.learn.model
        m.eval()
        cols = 4
        labels = listify(ifnone(label, self.labels))
        n = len(labels)
        lbl_rows = k
        naxs_lbl = k*cols
        rows = lbl_rows * n
        fig,axs = plt.subplots(rows, cols, figsize=(size*cols,size*rows))
        axs = axs.flatten()
        for ax in axs: ax.set_axis_off()
        for i,lbl in enumerate(labels):
            data = self.topk_iou(k, lbl, ascending=ascending)[lbl]
            idxs = [i for i,o in enumerate(self.d['labels']) if lbl in o]
            idxs = [idxs[i] for i in data['idxs']]
            lbl_axs = axs[i*naxs_lbl:(i+1)*naxs_lbl]
            for i,(idx,iou,true_box,pred_box,sal,det,mix) in enumerate(zip(idxs,data['ious'],data['true_boxes'],data['pred_boxes'],
                                                                           data['saliency'],data['detection'],data['mix'])):
                taxs = lbl_axs[i*cols:(i+1)*cols]
                x = self.data.valid_ds.x[idx]
                x.show(ax=taxs[0])
                taxs[0].set_title(f'{lbl}\niou={iou:.4f}')
                sal.show(taxs[1], sz=det.shape, vmin=0, vmax=1).set_title('saliency')
                det.show(taxs[2]).set_title('detection branch')
                mix.show(taxs[3]).set_title('mix')
                for i,tax in enumerate(taxs):
                    pred_box.draw(tax, color='green', fill=i==0)
                    true_box.draw(tax, color='white', fill=i==0)

        return axs

    def get_report(self, threadholds:Floats=[.1,.2,.3,.4,.5,.6,.7])->DataFrame:
        series = []
        for th in threadholds:
            d = {'m': th}
            for lbl in self.labels:
                ious = self.ious[lbl]
                d[lbl] = ious.ge(th).float().mean().item()

            series.append(d)

        d = {'m': 'IoU'}
        for lbl in self.labels:
            ious = self.ious[lbl]
            d[lbl] = ious.mean().item()

        series.append(d)
        return DataFrame(series).assign(avg=lambda d: d.mean(axis=1)).round(4)

    def plot_iou_dist(self, cols:int=4, size:int=4)->Collection[plt.Axes]:
        import seaborn as sns

        n = len(self.labels)
        rows = math.ceil(n/cols)
        fig,axs = plt.subplots(rows, cols, figsize=(size*cols,size*rows))
        for lbl,ax in zip(self.labels,axs.flatten()):
            ious = tensor(self.get_label(lbl, 'ious'))
            sns.distplot(ious, ax=ax)
            ax.set_title(lbl)

        return axs
