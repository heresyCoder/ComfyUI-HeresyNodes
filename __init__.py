from .nodes import SaveWanVideoLatent, LoadWanVideoLatent,  GetDateTimeStringUpd, SaveImageBatchNoMeta, RaiseExceptionMessage, LoadSingleImage, DiffModelSelect, CLIPModelSelector, LoadImagesByCond, LoadImageFolders, GetSubfolders, StitchImgMulti, DeleteNumberedFolders, SaveDictLatentAndImages

NODE_CLASS_MAPPINGS = {
    "SaveWanVideoLatent": SaveWanVideoLatent,
    "LoadWanVideoLatent": LoadWanVideoLatent,
    "GetDateTimeStringUpd": GetDateTimeStringUpd,
    "SaveImageBatchNoMeta": SaveImageBatchNoMeta,
    "RaiseExceptionMessage": RaiseExceptionMessage,
    "LoadSingleImage": LoadSingleImage,
    "DiffModelSelect": DiffModelSelect,
    "CLIPModelSelector": CLIPModelSelector,
    "LoadImagesByCond": LoadImagesByCond,
    "LoadImageFolders": LoadImageFolders,
    "GetSubfolders": GetSubfolders,
    "StitchImgMulti": StitchImgMulti,
    "DeleteNumberedFolders": DeleteNumberedFolders,
    "SaveDictLatentAndImages": SaveDictLatentAndImages,   
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveWanVideoLatent": "⛧Save Wan Video Latent (For Loops)",
    "LoadWanVideoLatent": "⛧Load Wan Video Latent",
    "GetDateTimeStringUpd": "⛧Get Date Time String (Updatable)",
    "SaveImageBatchNoMeta": "⛧Save Image Batch (For Loops)",
    "RaiseExceptionMessage": "⛧Raise Exception Message",
    "LoadSingleImage": "⛧Load Single Image",
    "DiffModelSelect": "⛧Diffusion Model Selector",
    "CLIPModelSelector": "⛧CLIP Model Selector",
    "LoadImagesByCond": "⛧Load Images Conditional",
    "LoadImageFolders": "⛧Load Image Folders",
    "GetSubfolders": "⛧Get Subfolders",
    "StitchImgMulti": "⛧Stitch Images Multi",
    "DeleteNumberedFolders": "⛧Delete Numbered Folders",
    "SaveDictLatentAndImages": "⛧Save Latent And Images",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']