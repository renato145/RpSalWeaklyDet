from fastai.vision import *
from .core import CustomObjectItemList, multiclass_bb_pad_collate
from ..utils import delegates

@delegates(get_transforms)
def get_chestxray8(path:PathOrStr, bs:int, img_sz:int, valid_only_bbx:bool=False, tfms:bool=True, convert_mode:str='RGB',
                   normalize:bool=True, norm_stats:Tuple[Floats, Floats]=imagenet_stats, processor:Optional[Callable]=None,
                   **kwargs:Any)->DataBunch:
    '''
    TODO
    '''
    path = Path(path)
    df = pd.read_pickle(path / 'full_ds_bbx.pkl')
    df['is_valid'] = df.set!='Train'
    if valid_only_bbx: df = df[(df.set=='Train') | df.bbx]

    if processor is not None: df = processor(df)

    lbl_dict = df[['file','label']].set_index('file')['label'].to_dict()
    def bbox_label_func(fn:str)->list: return lbl_dict[Path(fn).name]

    lbls = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
            'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    src = (CustomObjectItemList.from_df(df, path / 'images', cols='file', convert_mode=convert_mode)
                               .split_from_df('is_valid')
                               .label_from_func(bbox_label_func, classes=lbls))

    if tfms: src = src.transform(get_transforms(**kwargs), size=img_sz, tfm_y=True)

    data =  src.databunch(bs=bs, collate_fn=multiclass_bb_pad_collate)
    if normalize: data = data.normalize(stats=norm_stats)

    return data
