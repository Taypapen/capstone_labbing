import os
import glob
import torch
import numpy as np
import sys
import cv2
import random

class Dataset(object):
    def __init__(self, data_partition, data_dir, label_fp, preprocessing_func=None, data_suffix='.npz'):
        self._data_partition = data_partition
        self._data_dir = data_dir
        self._data_suffix = data_suffix

        #self._getting_data = True

        self._label_fp = label_fp
        self.label_idx = -3
        self.preprocessing_func = preprocessing_func

        self._data_files = []

        self.load_dataset()

    def load_dataset(self):
        #for now, labels are not read from file
        #self._labels = read_txt_lines(self._label_fp)
        self._labels = self._label_fp

        #add files to self._data_files
        self._get_files_for_part()

        # -- from self._data_files to self.list
        self.list = dict()
        for i, x in enumerate(self._data_files):
            label = self._get_label_from_path(x)
            self.list[i] = [x, self._labels.index(label)]

        print('Partition {} loaded'.format(self._data_partition))

    def _get_label_from_path(self, x):
        return x.split('/')[self.label_idx]

    def _get_files_for_part(self):

        dir_fp = self._data_dir
        if not dir_fp:
            return

        #get npy files
        search_npz = os.path.join(dir_fp, '*', self._data_partition, '*.npz')
        self._data_files.extend(glob.glob(search_npz))

        #remove examples if label not being used
        self._data_files = [f for f in self._data_files if f.split('/')[self.label_idx] in self._labels]

    def load_data(self, filename):
        try:
            if filename.endswith('.npz'):
                return np.load(filename)['data']
            else:
                return np.load(filename)
        except IOError:
            print('Error when reading file: {}'.format(filename))
            sys.exit()

    def __getitem__(self, idx):
        #try:
        raw_data = self.load_data(self.list[idx][0])
        preprocess_data = self.preprocessing_func(raw_data)
        label = self.list[idx][1]
        return preprocess_data, label
        #except KeyError:
        #    print("Data loaded up to idx {}".format(str(idx)))

    def __len__(self):
        return len(self._data_files)

#%%
class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class BgrToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_BGR2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)

class CenterCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

class RandomCrop(object):
    """Crop the given image to size randomly from edges
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    """Flip image horizontally.
    """

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames
#%%
def preprocess_creation():
    preprocessing = {}

    crop_size = (88,88)
    (mean, std) = (0.421, 0.165)

    preprocessing['train'] = Compose([
                                    BgrToGray(),
                                    Normalize( 0.0,255.0 ),
                                    RandomCrop(crop_size),
                                    HorizontalFlip(0.5),
                                    Normalize(mean, std) ])
    preprocessing['val'] = Compose([
                                    BgrToGray(),
                                    Normalize( 0.0,255.0 ),
                                    CenterCrop(crop_size),
                                    Normalize(mean, std) ])
    preprocessing['test'] = preprocessing['val']

    return preprocessing


def collate_fn(batch):
    if len(batch) == 1:
        data, lengths, labels_np, = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])
        data = torch.FloatTensor(data)
        lengths = [data.size(1)]

    if len(batch) > 1:
        data_list, lengths, labels_np = zip(*[(a, a.shape[0], b) for (a, b) in sorted(batch, key=lambda x: x[0].shape[0], reverse=True)])

        if data_list[0].ndim == 3:
            max_len, h, w = data_list[0].shape  # since it is sorted, the longest video is the first one
            data_np = np.zeros((len(data_list), max_len, h, w))
        elif data_list[0].ndim == 1:
            max_len = data_list[0].shape[0]
            data_np = np.zeros((len(data_list), max_len))
        for idx in range(len(data_np)):
            data_np[idx][:data_list[idx].shape[0]] = data_list[idx]
        data = torch.FloatTensor(data_np)
    labels = torch.LongTensor(labels_np)
    return data, lengths, labels


def dataloaders(data_dir, label_fp, batch_size, workers=1):
    preprocessing = preprocess_creation()

    #create datasets for train,test,val partitions
    datasets = {partition: Dataset(data_partition=partition,
                                   data_dir=data_dir,
                                   label_fp=label_fp,
                                   preprocessing_func=preprocessing[partition],
                                   data_suffix='.npz')
                for partition in ['train', 'test']}

    dset_loaders = {x: torch.utils.data.DataLoader(
                        datasets[x],
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        num_workers=workers,
                        worker_init_fn=np.random.seed(1)) for x in ['train', 'test']}
    return dset_loaders