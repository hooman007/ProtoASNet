import os
from random import randint
import warnings
import logging

import pandas as pd
from scipy.io import loadmat
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import RandomResizedCropVideo

from random import uniform

from src.data.video_transforms import RandomRotateVideo

# filter out pytorch user warnings for upsampling behaviour
warnings.filterwarnings("ignore", category=UserWarning)

class_labels = ["No AS", "Early AS", "Significant AS"]


def get_as_dataloader(config, split, mode):  # TODO modify to your own dataset
    """
    Uses the configuration dictionary to instantiate AS dataloaders

    Parameters
    ----------
    config : data configuration in dictionary format
    split : string, 'train'/'val'/'test' for which section to obtain
    mode : string, 'train'/'val'/'push'/'test' for setting augmentation/metadata ops

    Returns
    -------
    Training, validation or test dataloader with data arranged according to
    pre-determined splits

    """
    num_workers = config["num_workers"]
    bsize = config["batch_size"]
    transform = config["augmentation"]
    iterate_intervals = False
    if mode != "train":
        transform = False
        if mode != "push":
            iterate_intervals = config["iterate_intervals"]
        if config["frames"] == 1:
            bsize = 150

    dset = AorticStenosisDataset(
        **config,
        split=split,
        transform=transform,
        interval_iteration=iterate_intervals,
    )
    if mode == "train":
        if config["sampler"] == "AS":
            sampler_as = dset.class_sampler_AS()
            loader = DataLoader(dset, batch_size=bsize, sampler=sampler_as, num_workers=num_workers)
        else:  # random sampling
            loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=num_workers)
    else:
        loader = DataLoader(dset, batch_size=bsize, shuffle=False, num_workers=num_workers)
    return loader


class AorticStenosisDataset(Dataset):  # TODO modify to your own dataset
    def __init__(
        self,
        data_info_file: str,
        view: str = "plax",  # one of  psax, plax, all
        split: str = "train",
        sample_size=None,  # to load only the first "sample_size" cases of the dataframe (for quick epoch runs)
        transform: bool = False,
        transform_rotate_degrees: float = 10.0,
        transform_min_crop_ratio: float = 0.7,
        transform_time_dilation: float = 0.2,
        normalize: bool = False,
        frames: int = 16,
        img_size: int = 224,
        interval_iteration: bool = False,  # False = one 'unit' per video, True = multiple 'units' in sequence
        interval_unit: str = "cycle",  # image/second/cycle = get X images/seconds/cycles
        interval_quant: float = 1.0,  # X in images/seconds/cycles
        **kwargs,
    ):
        # read in the data directory CSV as a pandas dataframe
        dataset = pd.read_csv(data_info_file)

        if view in ("plax", "psax"):
            dataset = dataset[dataset["view"] == view]
        elif view != "all":
            raise ValueError(f"View should be plax/psax/all, got {view}")

        # Take train/test/val
        if split in ("train", "val", "test"):
            dataset = dataset[dataset["split"] == split]
        elif split != "all":
            raise ValueError(f"View should be train/val/test/all, got {split}")

        if sample_size is not None:
            dataset = dataset.sample(sample_size)

        # check the number of image/sub-video intervals we can get per video
        # and create a mapping between dataset entries and intervals
        self.interval_iteration = interval_iteration
        self.interval_unit = interval_unit
        self.interval_quant = interval_quant
        if frames == 1:
            assert interval_unit == "image", "For drawing 1 frame from dataloader, interval_unit must be image"
            assert frames == interval_quant, "For drawing 1 frame from dataloader, interval_quant must also be 1"
        if self.interval_iteration:
            dataset, dataset_intervals = compute_intervals(dataset, interval_unit, interval_quant)
            self.dataset_intervals = dataset_intervals
        else:
            dataset, _ = compute_intervals(dataset, interval_unit, interval_quant)

        self.dataset = dataset

        self.frames = frames
        self.resolution = (img_size, img_size)

        self.transform = None
        self.transform_time_dilation = 0.0
        if transform:
            self.transform = Compose(
                [
                    RandomResizedCropVideo(size=self.resolution, scale=(transform_min_crop_ratio, 1)),
                    RandomRotateVideo(degrees=transform_rotate_degrees),
                ]
            )
            self.transform_time_dilation = transform_time_dilation
        self.normalize = normalize

    def class_sampler_AS(self):
        """
        returns samplers (WeightedRandomSamplers) based on frequency of the AS class occurring
        """
        labels_as = self.dataset.apply(lambda x: x.as_label, axis=1).values
        class_sample_count_as = self.dataset.as_label.value_counts().to_numpy()
        weight_as = 1.0 / class_sample_count_as
        samples_weight_as = weight_as[labels_as]
        sampler_as = WeightedRandomSampler(samples_weight_as, len(samples_weight_as))
        return sampler_as

    def __len__(self) -> int:
        """
        iterative interval mode uses an "expanded" version of the dataset
        where each interval of a video can be considered as a separate data
        instance, thus the length of the dataset depends on the interval mode
        """
        if self.interval_iteration:
            return len(self.dataset_intervals)
        else:
            return len(self.dataset)

    @staticmethod
    def get_random_interval(vid_length, length):
        if length > vid_length:
            return 0, vid_length
        else:
            start = randint(0, vid_length - length)
            return start, start + length

    # expands one channel to 3 color channels, useful for some pretrained nets
    @staticmethod
    def gray_to_gray3(in_tensor):
        # in_tensor is 1xTxHxW
        return in_tensor.expand(3, -1, -1, -1)

    # normalizes pixels based on pre-computed mean/std values
    @staticmethod
    def bin_to_norm(in_tensor):
        """
        normalizes the input tensor
        :param in_tensor: needs to be already in range of [0,1]
        """
        # in_tensor is 1xTxHxW
        m = 0.099
        std = 0.171
        return (in_tensor - m) / std

    def _get_item_from_info(self, data_info, window_start, window_end, interval_idx=0):
        """
        General method to get an image and apply tensor transformation to it

        Parameters
        ----------
        data_info : pd.DataFrame
            1-row dataframe of item to retrieve
        window_start : int
            frame of the start of the interval to retrieve
        window_end : int
            frame of the end of the interval to retrieve (non-inclusive)
        interval_idx : int
            index of the interval within the cine (eg. first interval is 0)

        Returns
        -------
        ret : 3xTxHxW tensor (if video) 3xHxW tensor (if image)
            representing image/video w standardized dimensionality
        """
        cine_original = loadmat(data_info["path"])["cine"]  # T_original xHxW
        cine = cine_original[window_start:window_end]  # T_window xHxW

        cine = resize(cine, (self.frames, *self.resolution))  # TxHxW, where T=1 if image, Note range becomes [0,1] here
        cine = torch.tensor(cine).unsqueeze(0)  # 1xTxHxW, where T=1 if image

        label_as = torch.tensor(data_info["as_label"])

        if self.transform:
            cine = self.transform(cine)
        if self.normalize:
            cine = self.bin_to_norm(cine)
        cine = self.gray_to_gray3(cine)  # shape = (3,T,H,W), where T=1 if image
        cine = cine.float()  # 3xTxHxW, where T=1 if image

        if self.frames == 1:
            cine = cine[:, 0]  # 3xTxHxW (for video), 3xHxW (for image)

        ret = {
            "filename": os.path.basename(data_info.path),
            "cine": cine,
            "target_AS": label_as,
            "interval_idx": interval_idx,
            "window_start": window_start,
            "window_end": window_end,
            "original_length": cine_original.shape[0],
        }
        return ret

    def __getitem__(self, item):
        """
        iterative interval mode uses an "expanded" version of the dataset
        where each interval of a video can be considered as a separate data
        instance, thus the length of the dataset depends on the interval mode
        """
        if self.interval_iteration:
            data_interval = self.dataset_intervals.iloc[item]
            video_id = data_interval["video_idx"]
            data_info = self.dataset.iloc[video_id]
            start_frame = data_interval["start_frame"]
            end_frame = data_interval["end_frame"]
            interval_idx = data_interval["interval_idx"]
        else:
            data_info = self.dataset.iloc[item]
            # determine a random window
            ttd = self.transform_time_dilation
            if self.interval_unit == "image":
                wsize = int(self.interval_quant)
            else:  # can slightly vary the window size
                wsize = max(int(data_info["window_size"] * uniform(1 - ttd, 1 + ttd)), 1)
            start_frame, end_frame = self.get_random_interval(data_info["frames"], wsize)
            interval_idx = 0

        return self._get_item_from_info(data_info, start_frame, end_frame, interval_idx)


def compute_intervals(df, unit, quantity):
    """
    Calculates the number of sub-videos from each video in the dataset
    Saves the frame window for each sub-video in a separate sheet

    Parameters
    ----------
    df : pd.DataFrame
        dataframe object containing frame rate, heart rate, etc.
    unit : str
        unit for interval retrieval, image/second/cycle
    quantity :
        quantity for interval retrieval,
        eg. 1.3 with "cycle" means each interval should be 1.3 cycles

    Returns
    -------
    df : pd.DataFrame
        updated dataframe containing num_intervals and window_size
    df_intervals: pd.DataFrame
        dataframe containing mapping between videos and window start/end frames

    """
    ms = df["frame_time"]
    hr = df["heart_rate"]
    if unit == "image":
        if int(quantity) < 1:
            raise ValueError("Must draw >= 1 image per video")
        df["window_size"] = int(quantity)
    elif unit == "second":
        df["window_size"] = (quantity * 1000 / ms).astype("int32")
    elif unit == "cycle":
        df["window_size"] = (quantity * 60000 / ms / hr).astype("int32")
    else:
        raise ValueError(f"Unit should be image/second/cycle, got {unit}")
    # if there are any window sizes of zero or less, raise an exception
    if len(df[df["window_size"] < 1]) > 0:
        raise Exception("Dataloader: Detected proposed window size of 0, exiting")

    df["num_intervals"] = (df["frames"] / df["window_size"]).astype("int32")

    video_idx, interval_idx, start_frame, end_frame = [], [], [], []
    for i in range(len(df)):
        video_info = df.iloc[i]
        if video_info["num_intervals"] == 0:
            video_idx.append(i)
            interval_idx.append(0)
            start_frame.append(0)
            end_frame.append(video_info["frames"])
        else:
            n_intervals = video_info["num_intervals"]
            w_size = video_info["window_size"]
            for j in range(n_intervals):
                video_idx.append(i)
                interval_idx.append(j)
                start_frame.append(j * w_size)
                end_frame.append((j + 1) * w_size)
    d = {
        "video_idx": video_idx,
        "interval_idx": interval_idx,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
    df_interval = pd.DataFrame.from_dict(d)

    return df, df_interval


if __name__ == "__main__":
    data_config = {   # TODO modify to your own dataset
        "name": "<dataset-name>",
        "data_info_file": "<CSV_NAME>",
        "sample_size": None,
        "sampler": "random",  # one of 'AS', 'random', 'bicuspid'
        # one of 'binary', 'all', 'not_severe', 'as_only', 'mild_moderate', 'moderate_severe'
        "view": "plax",  # one of  psax, plax, all
        "normalize": True,
        "augmentation": True,
        "img_size": 112,
        "frames": 32,
        "transform_min_crop_ratio": 0.7,
        "transform_rotate_degrees": 15,
        "batch_size": 2,
        "num_workers": 0,
        "iterate_intervals": True,
        "interval_unit": "cycle",
        "interval_quant": 1.0,
    }
    split = "train"
    dataloader = get_as_dataloader(data_config, split=split, mode=split)

    # dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    # print("len of dataset = {}".format(len(dataset)))
    logging.info("len of dataloader iteration = {}".format(len(dataloader)))
    # ################### Iterate over samples #########################

    data_iter = iter(dataloader)
    sample_dict = next(data_iter)  # sample shape = (N,3,T,H,W) for video, (N,3,H,W) for image

    logging.info(
        f"target_AS: {sample_dict['target_AS']}\n"
        f"Label shape : {sample_dict['target_AS'].shape} \n"
        f"filenames : {sample_dict['filename']} \n"
        f"cine shape: {sample_dict['cine'].shape}"
    )
