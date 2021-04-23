from datasets.data_set import PetDataset
import numpy as np
import os


# 数据装载
class GradImg(PetDataset):

    def __init__(self, data_dir, mode='train'):
        # self.image_size = IMAGE_SIZE
        self.mode = mode.lower()
        self.data_dir = data_dir

        assert self.mode in ['train', 'test', 'predict'], \
            "mode should be 'train' or 'test' or 'predict', but got {}".format(self.mode)

        self.train_images = []
        self.label_images = []
        self.grad_images = []
        list_file = os.path.join(self.data_dir, '{}.txt'.format(self.mode))
        with open(list_file, 'r') as f:
            for line in f.readlines():
                image, label, grad = line.strip().split('\t')
                self.train_images.append(image)
                self.label_images.append(label)
                self.grad_images.append(grad)

    def __getitem__(self, idx):
        train_image = self._load_img(self.train_images[idx])
        label_image = self._load_img(self.label_images[idx])
        grad_image = self._load_img(self.grad_images[idx])
        train_image = train_image[np.newaxis, :].astype('float32')
        label_image = label_image[np.newaxis, :].astype('float32')
        grad_image = grad_image[np.newaxis, :].astype('float32')
        return train_image, label_image, grad_image



