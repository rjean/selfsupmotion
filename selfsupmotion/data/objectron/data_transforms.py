import typing

import albumentations
import numpy as np
import PIL
import torchvision

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

    @staticmethod
    def _generate_obj_crops(sample: typing.Dict, crop_height: int):
        """
        This operation will crop all frames in a sequence based on the object center location
        in the first frame. This will allow the model to perceive some of the camera movement.
        """
        assert isinstance(sample, dict) and "IMAGE" in sample and "CENTROID_2D_IM" in sample
        assert len(sample["IMAGE"].shape) == 4 and sample["IMAGE"].shape[1:] == (640, 480, 3)
        assert len(sample["CENTROID_2D_IM"].shape) == 2 and sample["CENTROID_2D_IM"].shape[-1] == 2
        # get top-left/bottom-right coords for object of interest in first frame (0-th index)
        tl = (int(round(sample["CENTROID_2D_IM"][0, 0] - crop_height / 2)),
              int(round(sample["CENTROID_2D_IM"][0, 1] - crop_height / 2)))
        br = (tl[0] + crop_height, tl[1] + crop_height)
        # get crops one at a time for all frames in the seq, for all seqs in the minibatch
        output_crop_seq = []
        for frame_idx, frame in enumerate(sample["IMAGE"]):
            if thelper_available:
                crop = thelper.draw.safe_crop(image=frame, tl=tl, br=br)
            else:
                crop = selfsupmotion.data.utils.safe_crop(image=frame, tl=tl, br=br)
            output_crop_seq.append(crop)
        assert "OBJ_CROPS" not in sample
        sample["OBJ_CROPS"] = output_crop_seq
        return sample

    def __init__(
            self,
            crop_height: int = 320,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
            seed_wrap_augments: bool = False,
            use_hflip_augment: bool = False,
    ) -> None:
        self.crop_height = crop_height
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.seed_wrap_augments = seed_wrap_augments
        self.use_hflip_augment = use_hflip_augment

        augment_transforms = [albumentations.RandomResizedCrop(
            height=self.input_height,
            width=self.input_height,
            scale=(0.4, 1.0),  # @@@@ adjust if needed?
            ratio=(1.0, 1.0),  # @@@@ adjust if needed?
        )]
        if use_hflip_augment:
            augment_transforms.append(albumentations.HorizontalFlip(p=0.5))
        augment_transforms.extend([
            albumentations.ColorJitter(
                brightness=0.6 * self.jitter_strength,
                contrast=0.6 * self.jitter_strength,
                saturation=0.6 * self.jitter_strength,
                hue=0.2 * self.jitter_strength,
                p=0.8,
            ),
            albumentations.ToGray(p=0.2),
        ])
        if self.gaussian_blur:
            # @@@@@ TODO: check what kernel size is best? is auto good enough?
            #kernel_size = int(0.1 * self.input_height)
            #if kernel_size % 2 == 0:
            #    kernel_size += 1
            augment_transforms.append(albumentations.GaussianBlur(
                blur_limit=(3, 5),
                #blur_limit=kernel_size,
                #sigma_limit=???
                p=0.5,
            ))

        if self.seed_wrap_augments:
            assert thelper_available
            self.augment_transform = thelper.transforms.wrappers.SeededOpWrapper(
                operation=albumentations.Compose(augment_transforms),
                sample_kw="image",
            )
        else:
            self.augment_transform = albumentations.Compose(augment_transforms)

        self.convert_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # add online train transform of the size of global view
        self.online_augment_transform = albumentations.Compose([
            albumentations.RandomResizedCrop(
                height=self.input_height,
                width=self.input_height,
                scale=(0.5, 1.0),  # @@@@ adjust if needed?
            ),  # @@@@@@@@@ BAD W/O SEED WRAPPER?
            albumentations.HorizontalFlip(p=0.5),  # @@@@@@@@@ BAD W/O SEED WRAPPER?
        ])

    def __call__(self, sample):
        # first, add the object crops to the sample dict
        sample = self._generate_obj_crops(sample, self.crop_height)
        # now, for each crop, apply the seeded transform list
        output_crops = []
        shared_seed = np.random.randint(np.iinfo(np.int32).max)
        for crop_idx in range(len(sample["OBJ_CROPS"])):
            np.random.seed(shared_seed)  # the wrappers will use numpy to re-seed themselves internally
            if self.seed_wrap_augments:
                aug_crop = self.augment_transform(sample["OBJ_CROPS"][crop_idx])
            else:
                aug_crop = self.augment_transform(image=sample["OBJ_CROPS"][crop_idx])
            output_crops.append(self.convert_transform(PIL.Image.fromarray(aug_crop["image"])))
        # finally, add the 'transformed global view' (??)
        aug_global_crop = self.online_augment_transform(image=sample["OBJ_CROPS"][0])
        output_crops.append(self.convert_transform(PIL.Image.fromarray(aug_global_crop["image"])))
        return output_crops


class SimSiamFramePairEvalDataTransform(SimSiamFramePairTrainDataTransform):
    """
    Transforms for SimSiam + Objectron:

        _first_frame_object_center_crop(size=320)  (custom for Objectron specifically)
        Resize(input_height + 10, interpolation=3) (to fix test-time crop size discrepency)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    (note: the transform list is copied and adapted from the SimCLR transforms)
    """

    def __init__(
            self,
            crop_height: int = 320,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.,
    ):
        super().__init__(
            crop_height=crop_height,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        adjusted_precrop_size = int(self.input_height + 0.1 * self.input_height)
        self.online_augment_transform = albumentations.Compose([
            albumentations.Resize(adjusted_precrop_size, adjusted_precrop_size),
            albumentations.CenterCrop(self.input_height, self.input_height),
        ])
