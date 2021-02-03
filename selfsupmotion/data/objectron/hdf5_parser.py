"""Objectron dataset parser module.

This module contains dataset parsers used to load the HDF5 archive of the Objectron dataset.
See https://github.com/google-research-datasets/Objectron for more info.
"""

import os
import typing
import time

import cv2 as cv
import h5py
import numpy as np
import pickle
import pytorch_lightning
import torch.utils.data
import tqdm

try:
    import turbojpeg
    turbojpeg = turbojpeg.TurboJPEG()
except ImportError:
    turbojpeg = None

import selfsupmotion.data.objectron.data_transforms
import selfsupmotion.data.objectron.sequence_parser
import selfsupmotion.data.utils


class ObjectronHDF5SequenceParser(torch.utils.data.Dataset):
    """Objectron HDF5 dataset parser.

    This class can be used to parse the (non-official) Objectron HDF5. More specifically, it allows
    random access over all sequences of the original dataset. See the HDF5 extractor's module for
    information on what the HDF5 files contain.
    """

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            objects: typing.Optional[typing.Sequence[typing.AnyStr]] = None,  # default => use all
    ):
        self.hdf5_path = hdf5_path
        assert os.path.exists(self.hdf5_path), f"invalid dataset path: {self.hdf5_path}"
        all_objects = selfsupmotion.data.objectron.sequence_parser.ObjectronSequenceParser.all_objects
        if not objects:
            self.objects = all_objects
        else:
            assert all([obj in all_objects for obj in objects]), "invalid object name used in filter"
            self.objects = objects
        self.seq_name_map = {}
        with h5py.File(self.hdf5_path, mode="r") as fd:
            self.target_subsampl_rate = fd.attrs["target_subsampl_rate"]
            self.objectron_subset = fd.attrs["objectron_subset"]
            self.data_fields = fd.attrs["data_fields"]
            self.attr_fields = fd.attrs["attr_fields"]
            for object in self.objects:
                if object not in fd:
                    print(f"missing '{object}' group in dataset, skipping...")
                    continue
                for seq_id, seq_data in fd[object].items():
                    self.seq_name_map[len(self.seq_name_map)] = object + "/" + seq_id
        assert len(self.seq_name_map) > 0, "invalid subset selected for given archive"
        self.local_fd = None  # will be opened by worker on first getitem call

    def __len__(self):
        return len(self.seq_name_map)

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.seq_name_map), "sequence index oob"
        if self.local_fd is None:
            self.local_fd = h5py.File(self.hdf5_path, mode="r")
        return self.local_fd[self.seq_name_map[idx]]


class ObjectronHDF5FrameTupleParser(ObjectronHDF5SequenceParser):
    """Objectron HDF5 dataset parser.

    This class can be used to parse the (non-official) Objectron HDF5. More specifically, it allows
    random access over frame tuples of the original dataset. See the HDF5 extractor's module for
    information on what the HDF5 files contain.

    What's a tuple in this context, you ask? Well, it's a series of consecutive frames (which
    might already be subsampled in the HDF5 writer) that are also separated by `frame_offset`
    extra frames. Consecutive tuples are then also separated by `tuple_offset` frames.
    """

    # @@@@ TODO in the future: create pairs on smallest pt reproj errors?

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            objects: typing.Optional[typing.Sequence[typing.AnyStr]] = None,  # default => use all
            tuple_length: int = 2,  # by default, we will create pairs of frames
            frame_offset: int = 1,  # by default, we will create tuples from consecutive frames
            tuple_offset: int = 2,  # by default, we will skip a frame between tuples
            _target_fields: typing.Optional[typing.List[typing.AnyStr]] = None,
            _transforms: typing.Optional[typing.Any] = None,
    ):
        cache_hash = selfsupmotion.data.utils.get_params_hash(
            {k: v for k, v in vars().items() if not k.startswith("_") and k != "self"})
        super().__init__(hdf5_path=hdf5_path, objects=objects)
        assert tuple_length >= 1 and frame_offset >= 1 and tuple_offset >= 1, "invalid tuple params"
        self.tuple_length = tuple_length
        self.frame_offset = frame_offset
        self.tuple_offset = tuple_offset
        self.target_fields = self.data_fields if not _target_fields else _target_fields
        self.transforms = _transforms
        cache_path = os.path.join(os.path.dirname(hdf5_path), cache_hash + ".pkl")
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as fd:
                self.frame_pair_map = pickle.load(fd)
        else:
            self.frame_pair_map = self._fetch_tuple_metadata()
            with open(cache_path, "wb") as fd:
                pickle.dump(self.frame_pair_map, fd)

    def _fetch_tuple_metadata(self):
        tuple_map = {}
        with h5py.File(self.hdf5_path, mode="r") as fd:
            for object in self.objects:
                if object not in fd:
                    continue
                progress_bar = tqdm.tqdm(
                    fd[object].items(), total=len(fd[object]),
                    desc=f"pre-fetching tuple metadata for '{object}'"
                )
                for seq_id, seq_data in progress_bar:
                    seq_len = len(seq_data["IMAGE"])
                    if seq_len <= self.frame_offset:  # skip, can't create frame tuples
                        continue
                    # note: the frame idx here is not the real index if the sequence was subsampled
                    tuple_start_idx = 0
                    while tuple_start_idx < seq_len:
                        candidate_tuple_idxs = []
                        for tuple_idx_offset in range(self.tuple_length):
                            curr_frame_idx = tuple_start_idx + tuple_idx_offset * self.frame_offset
                            if curr_frame_idx >= seq_len:
                                break  # if we're oob for any element in the tuple, break it off
                            if not self._is_frame_valid(seq_data, curr_frame_idx):
                                break  # if an element has a bad frame, break it off
                            candidate_tuple_idxs.append(curr_frame_idx)
                        if len(candidate_tuple_idxs) == self.tuple_length:  # if it's all good, keep it
                            tuple_map[len(tuple_map)] = \
                                (object + "/" + seq_id, tuple(candidate_tuple_idxs))
                        tuple_start_idx += self.tuple_offset
        return tuple_map

    @staticmethod
    def _is_frame_valid(dataset, frame_idx):
        # to keep things light and fast (enough), we'll only look at object centroids
        im_coords = dataset["CENTROID_2D_IM"][frame_idx]
        # in short, if any centroid is outside the image frame by more than its size, it's bad
        # (this will catch crazy-bad object coordinates that would lead to insanely big crops)
        return -640 < im_coords[0] < 1280 and -480 < im_coords[1] < 960

    def __len__(self):
        return len(self.frame_pair_map)

    @staticmethod
    def _decode_jpeg(data):
        if turbojpeg is not None:
            image = turbojpeg.decode(data)
        else:
            image = cv.imdecode(data, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        assert 0 <= idx < len(self.frame_pair_map), "frame pair index oob"
        if self.local_fd is None:
            self.local_fd = h5py.File(self.hdf5_path, mode="r")
        meta = self.frame_pair_map[idx]
        seq_data = self.local_fd[meta[0]]
        sample = {
            field: np.stack([
                self._decode_jpeg(seq_data[field][frame_idx])
                if field == "IMAGE" else seq_data[field][frame_idx]
                for frame_idx in meta[1]
            ]) for field in self.target_fields
        }
        sample = {"SEQ_ID": meta[0], **sample}
        if self.transforms is not None:
            sample = self.transforms(sample)
        class_label = self.objects.index(meta[0].split("/")[0])
        return sample, class_label


class ObjectronFramePairDataModule(pytorch_lightning.LightningDataModule):

    name = "objectron_fp"

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            objects: typing.Optional[typing.Sequence[typing.AnyStr]] = None,  # default => use all
            tuple_length: int = 2,
            frame_offset: int = 1,
            tuple_offset: int = 2,
            input_height: int = 224,
            gaussian_blur: bool = True,
            jitter_strength: float = 1.0,
            valid_split_ratio: float = 0.1,
            num_workers: int = 8,
            batch_size: int = 256,
            seed: int = 1337,
            shuffle: bool = False,
            pin_memory: bool = False,
            drop_last: bool = False,
            *args: typing.Any,
            **kwargs: typing.Any,
    ):
        super().__init__(*args, **kwargs)
        self.image_size = input_height
        self.dims = (3, input_height, input_height)
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.valid_split_ratio = valid_split_ratio
        self.hdf5_path = hdf5_path
        self.objects = objects
        self.tuple_length = tuple_length
        self.frame_offset = frame_offset
        self.tuple_offset = tuple_offset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # create temp dataset to get sample count
        dataset = ObjectronHDF5FrameTupleParser(
            hdf5_path=self.hdf5_path,
            objects=self.objects,
            tuple_length=self.tuple_length,
            frame_offset=self.frame_offset,
            tuple_offset=self.tuple_offset,
        )
        # note: split below is not ideal for current frame-pair parser, it leaks across sequences
        self.valid_sample_count = int(len(dataset) * self.valid_split_ratio)
        self.train_sample_count = len(dataset) - self.valid_sample_count
        assert self.train_sample_count > 0 and self.valid_sample_count > 0

    def _create_dataloader(
            self,
            train: bool = True,
            transforms: typing.Optional[typing.Any] = None,
    ) -> torch.utils.data.DataLoader:
        # @@@@ TODO: transforms should be passed to the dataset parser
        dataset = ObjectronHDF5FrameTupleParser(
            hdf5_path=self.hdf5_path,
            objects=self.objects,
            tuple_length=self.tuple_length,
            frame_offset=self.frame_offset,
            tuple_offset=self.tuple_offset,
            _target_fields=["IMAGE", "CENTROID_2D_IM"],
            _transforms=transforms,
        )
        # note: split below is not ideal for current frame-pair parser, it leaks across sequences
        dataset_train, dataset_valid = torch.utils.data.random_split(
            dataset, [self.train_sample_count, self.valid_sample_count],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = torch.utils.data.DataLoader(
            dataset_train if train else dataset_valid,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self._create_dataloader(
            train=True,
            transforms=self.train_transform() if self.train_transforms is None else self.train_transforms,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self._create_dataloader(
            train=False,
            transforms=self.val_transform() if self.val_transforms is None else self.val_transforms,
        )

    def train_transform(self):
        return selfsupmotion.data.objectron.data_transforms.SimSiamFramePairTrainDataTransform(
            input_height=self.image_size,
            gaussian_blur=self.gaussian_blur,
            jitter_strength=self.jitter_strength,
        )

    def val_transform(self):
        return selfsupmotion.data.objectron.data_transforms.SimSiamFramePairEvalDataTransform(
            input_height=self.image_size,
            gaussian_blur=self.gaussian_blur,
            jitter_strength=self.jitter_strength,
        )


if __name__ == "__main__":
    data_path = "/wdata/datasets/objectron/"
    hdf5_path = data_path + "extract_s5_raw.hdf5"

    batch_size = 1
    max_iters = 50
    dm = ObjectronFramePairDataModule(
        hdf5_path=hdf5_path,
       # frame_offset=2,
        input_height=224,
        gaussian_blur=True,
        jitter_strength=1.0,
        batch_size=batch_size,
        num_workers=2,
    )
    assert dm.train_sample_count > 0

    loader = dm.train_dataloader()
    norm_std = np.asarray([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    norm_mean = np.asarray([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    display = batch_size == 1

    iter = 0
    init_time = time.time()
    for batch in tqdm.tqdm(loader, total=len(loader)):
        frames, label = batch
        if display:
            display_frames = []
            for frame_idx, frame in enumerate(frames):
                # de-normalize and ready image for opencv display to show the result of transforms
                frame = (((frame.squeeze(0).numpy().transpose((1, 2, 0)) * norm_std) + norm_mean) * 255).astype(np.uint8)
                display_frames.append(frame)
            cv.imshow(
                f"frames",
                cv.resize(
                    cv.hconcat(display_frames),
                    dsize=(-1, -1), fx=4, fy=4, interpolation=cv.INTER_NEAREST,
                )
            )
            cv.waitKey(0)
        iter += 1
        if max_iters is not None and iter > max_iters:
            break
    print(f"all done in {time.time() - init_time} seconds")
