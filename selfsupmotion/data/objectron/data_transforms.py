import typing

import albumentations
from albumentations.augmentations import transforms
from albumentations.augmentations.functional import normalize
import numpy as np
import PIL
import torchvision
import random
import albumentations.pytorch

try:
    import thelper
    thelper_available = True
except ImportError:
    thelper_available = False

import selfsupmotion.data.utils


class SimSiamFramePairTrainDataTransform(object):
    """
    Transforms for SimSiam + Objectron:

        _generate_obj_crops(size=320)              (custom for Objectron specifically)
        RandomResizedCrop(size=self.input_height)  (grabs a fixed-size subregion to encode)
        RandomHorizontalFlip()                     (this and following ops apply to all frames)
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    (note: the transform list is copied and adapted from the SimCLR transforms)
    """

    #@staticmethod
    def _generate_obj_crops(self, sample: typing.Dict, crop_height: typing.Union[int, str]):
        """
        This operation will crop all frames in a sequence based on the object center location
        in the first frame. This will allow the model to perceive some of the camera movement.
        """
        assert isinstance(sample, dict) and "IMAGE" in sample and "CENTROID_2D_IM" in sample
        assert len(sample["IMAGE"].shape) == 4 and sample["IMAGE"].shape[1:] == (640, 480, 3)
        assert len(sample["CENTROID_2D_IM"].shape) == 2 and sample["CENTROID_2D_IM"].shape[-1] == 2
        assert self.crop_strategy in ["centroid","bbox"]
        # get top-left/bottom-right coords for object of interest in first frame (0-th index)
        if self.crop_strategy=="centroid":
            if isinstance(crop_height, int):
                tl = (int(round(sample["CENTROID_2D_IM"][0, 0] - crop_height / 2)),
                      int(round(sample["CENTROID_2D_IM"][0, 1] - crop_height / 2)))
                br = (tl[0] + crop_height, tl[1] + crop_height)
            else:
                assert crop_height == "auto", "unexpected crop height arg"
                base_pts = np.asarray([pt for pt in sample["POINTS"][0]])
                real_tl = (base_pts[:, 0].min(), base_pts[:, 1].min())
                real_br = (base_pts[:, 0].max(), base_pts[:, 1].max())
                max_size = max(real_br[0] - real_tl[0], real_br[1] - real_tl[1]) * 1.2  # 20% extra
                tl = (int(round(sample["CENTROID_2D_IM"][0, 0] - max_size / 2)),
                      int(round(sample["CENTROID_2D_IM"][0, 1] - max_size / 2)))
                br = (int(round(tl[0] + max_size)), int(round(tl[1] + max_size)))
        elif self.crop_strategy=="bbox":
            tl = (int(sample["POINTS"][0,:,0].min()), int(sample["POINTS"][0,:,1].min()))
            br = (int(sample["POINTS"][0,:,0].max()), int(sample["POINTS"][0,:,1].max()))
        else:
            raise ValueError(f"Invalid cropping stragegy: {self.crop_strategy}")
        # get crops one at a time for all frames in the seq, for all seqs in the minibatch
        if tl==br: #should not happen!
            print(f"Annotation error on {sample['UID']}, moving on!")
            tl = br[0]-64, br[1]-64 #Arbirary crop, just to avoid crashing!
        output_crop_seq = []
        output_keypoints = []
        for frame_idx, (frame, kpts) in enumerate(zip(sample["IMAGE"], sample["POINTS"])):
           
            if thelper_available:
                crop = thelper.draw.safe_crop(image=frame, tl=tl, br=br)
            else:
                crop = selfsupmotion.data.utils.safe_crop(image=frame, tl=tl, br=br)
            output_crop_seq.append(crop)
            if "POINTS" in sample:
                offset_coords = (tl[0], tl[1], 0, 0)
                output_keypoints.append(np.subtract(sample["POINTS"][frame_idx], offset_coords))
        assert "OBJ_CROPS" not in sample
        sample["OBJ_CROPS"] = output_crop_seq
        if output_keypoints:
            sample["POINTS"] = output_keypoints
        for i, obj_crop in enumerate(sample["OBJ_CROPS"]):
            if obj_crop.shape[0]<1 or obj_crop.shape[1]<1 or obj_crop.shape[2]!=3:
                print(f"Unable to take crop on {sample['UID']}, moving on!")
                sample["OBJ_CROPS"][i] = np.zeros((64,64,3), dtype=np.uint8)
            assert len(sample["OBJ_CROPS"][i]) > 0, "Unable to crop image!" #Don't allow return empty object!
        return sample

    def __init__(
            self,
            crop_height: typing.Union[int, typing.AnyStr] = 320,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
            seed_wrap_augments: bool = False,
            use_hflip_augment: bool = False,
            drop_orig_image: bool = True,
            crop_scale: typing.Tuple[float, float] = (0.2, 1.0),
            crop_ratio: typing.Tuple[float, float] = (1.0, 1.0),
            shared_transform = True,
            augmentation = True, #Will be used to disable augmentation on inference / validation.
            crop_strategy = "centroid",
            sync_hflip= False,
            coords = None,
    ) -> None:
        self.crop_height = crop_height
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.seed_wrap_augments = seed_wrap_augments
        self.use_hflip_augment = use_hflip_augment
        self.drop_orig_image = drop_orig_image
        self.shared_transform = shared_transform
        self.enable_augmentation= augmentation
        self.crop_strategy = crop_strategy
        self.sync_hflip=sync_hflip
        self.pixel_transform = albumentations.Normalize()
        self.coords = coords

        pixel_transforms = [
                albumentations.ColorJitter(
                    brightness=0.4 * self.jitter_strength,
                    contrast=0.4 * self.jitter_strength,
                    saturation=0.4 * self.jitter_strength,
                    hue=0.1 * self.jitter_strength,
                    p=0.8,
                ),
                albumentations.ToGray(p=0.2),
                albumentations.Normalize(),
            ]

        gaussian_blur = albumentations.GaussianBlur(
                    blur_limit=(3, 5),
                    #blur_limit=kernel_size,
                    #sigma_limit=???
                    p=0.5,
                )

        bbox_transforms = [
                albumentations.LongestMaxSize(
                    max_size=224
                ),
                albumentations.PadIfNeeded(
                    min_height=224,
                    min_width=224,
                    border_mode=0,
                )
            ]
        assert self.crop_strategy in ["centroid","bbox"]

        if self.enable_augmentation:
            self.pixel_transform = albumentations.Compose(pixel_transforms)
            geometric_transforms = [
                albumentations.RandomResizedCrop(
                    height=self.input_height,
                    width=self.input_height,
                    scale=crop_scale,
                    ratio=crop_ratio,
                ),
            ]
            if self.crop_strategy=="bbox":
                geometric_transforms = bbox_transforms + geometric_transforms
            if use_hflip_augment:
                geometric_transforms.append(albumentations.HorizontalFlip(p=0.5))
            #augment_transforms.extend(pixel_transforms)
            if self.gaussian_blur:
                geometric_transforms.append(gaussian_blur)
        else:
            geometric_transforms = bbox_transforms
        if self.seed_wrap_augments:
            assert thelper_available
            self.geometric_transform = thelper.transforms.wrappers.SeededOpWrapper(
                operation=albumentations.Compose(geometric_transforms),
                sample_kw="image",
            )
        else:
            self.geometric_transform = albumentations.Compose(geometric_transforms)

        self.convert_transform = albumentations.Compose([
            albumentations.pytorch.transforms.ToTensorV2(),
        ])
        #    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #])
        #self.convert_transform = albumentations.pytorch.transforms.ToTensor(normalize=
        #    {"mean":[0.485, 0.456, 0.406], "std":[0.229, 0.224, 0.225]}
        #)

        # add online train transform of the size of global view
        self.online_augment_transform = albumentations.Compose([
            albumentations.RandomResizedCrop(
                height=self.input_height,
                width=self.input_height,
                scale=(0.5, 1.0),  # @@@@ adjust if needed?
            ),  # @@@@@@@@@ BAD W/O SEED WRAPPER?
            albumentations.HorizontalFlip(p=0.5),  # @@@@@@@@@ BAD W/O SEED WRAPPER?
        ])
        self.sync_hflip_transform =  albumentations.HorizontalFlip(p=1)

    def __call__(self, sample):
        assert isinstance(sample, dict)
        # first, add the object crops to the sample dict
        sample = self._generate_obj_crops(sample, self.crop_height)
        # now, for each crop, apply the seeded transform list
        output_crops, output_keypoints = [], []
        shared_seed = np.random.randint(np.iinfo(np.int32).max)
        
        #if self.enable_augmentation: #Might be disabled for evalution or validation.
        flip =  random.choice([True,False]) 

        for crop_idx in range(len(sample["OBJ_CROPS"])):
            if self.shared_transform:
                np.random.seed(shared_seed)  # the wrappers will use numpy to re-seed themselves internally
           
            if self.seed_wrap_augments:
                assert "POINTS" not in sample, "missing impl"
                aug_crop = self.geometric_transform(sample["OBJ_CROPS"][crop_idx])
            else:
                #TODO: Add other channels.
                if self.pixel_transform:
                    pixels = self.pixel_transform(
                        image=sample["OBJ_CROPS"][crop_idx],
                    )
                    sample["OBJ_CROPS"][crop_idx]=pixels["image"]

                crop = sample["OBJ_CROPS"][crop_idx]
                #crop = self.convert_transform(image=crop)["image"]
                #Based on CoordConv ideas. This will feed additional coordinates 
                #chanels to the first layer of the ResNet50. 
                if self.coords=="xy":
                    w = crop.shape[0]
                    h = crop.shape[1]
                    posx = np.tile(np.arange(0,w),(h,1)).T / w -0.5
                    posy = np.tile(np.arange(0,h),(w,1)) / h -0.5
                    crop = np.dstack([crop,posx, posy])

                aug_crop = self.geometric_transform(
                    image=crop,
                    keypoints=sample["POINTS"][crop_idx],
                    # the "xy" format somehow breaks when we have 2-coord kpts, this is why we pad to 4...
                    keypoint_params=albumentations.KeypointParams(format="xysa", remove_invisible=False),
                )

                if self.sync_hflip and flip:
                    aug_crop = self.sync_hflip_transform(
                        image=aug_crop["image"],
                        keypoints=aug_crop["keypoints"],
                        keypoint_params=albumentations.KeypointParams(format="xysa", remove_invisible=False)
                    )
                output_keypoints.append(aug_crop["keypoints"])
                aug_crop = self.convert_transform(
                        image=aug_crop["image"],
                        keypoints=aug_crop["keypoints"],
                        keypoint_params=albumentations.KeypointParams(format="xysa", remove_invisible=False)
                    )
                sample["OBJ_CROPS"][crop_idx] = aug_crop["image"]
            #output_crops.append(self.convert_transform(PIL.Image.fromarray(aug_crop["image"])))
        #sample["OBJ_CROPS"] = output_crops
        # finally, scrap the dumb padding around the 2d keypoints
        sample["POINTS"] = [pts for pts in np.asarray(output_keypoints)[..., :2].astype(np.float32)]
        if self.drop_orig_image:
            del sample["IMAGE"]
            del sample["CENTROID_2D_IM"]
        return sample


class SimSiamFramePairEvalDataTransform(SimSiamFramePairTrainDataTransform):
    """
    Transforms for SimSiam + Objectron:

        _first_frame_object_center_crop(size=320)  (custom for Objectron specifically)
        Resize(input_height + 10, interpolation=3) (to fix test-time crop size discrepency)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    (note: the transform list is copied and adapted from the SimCLR transforms)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # replace online transform with eval time transform
        adjusted_precrop_size = int(self.input_height + 0.1 * self.input_height)
        self.online_augment_transform = albumentations.Compose([
            albumentations.Resize(adjusted_precrop_size, adjusted_precrop_size),
            albumentations.CenterCrop(self.input_height, self.input_height),
        ])
