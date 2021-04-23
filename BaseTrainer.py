import sys 
sys.path.append('/home/aistudio/external-libraries')
import paddle 
import paddle.nn.functional as F
from myutiles import *
import numpy as np
import os
# import nibabel as nib
import SimpleITK as sitk


class BaseTrainer:

    def __init__(self, network, data_dir, model_dir, model_name, log_dir, datasets):
        self.network = network()
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.log_dir = log_dir
        self.datasets = datasets
        try:
            self.train_datasets = datasets(self.data_dir)
        except:
            self.train_datasets = None
        try:
            self.test_datasets = datasets(self.data_dir, 'test')
        except:
            self.test_datasets = None
        
        self.patch_shape = (128, 128, 128)
        self.stride_shape = (48, 48, 48)
        self.img_shape = (181, 216, 181)


    def prepare_data(self):
        dirs = ['trainMR', 'trainCT', 'testMR', 'testCT']
        for d in dirs:
            path = os.path.join(self.data_dir, d)
            try:
                os.mkdir(path)
            except:
                pass

        train_mr_files, train_ct_files = [], []
        test_mr_files, test_ct_files = [], []
        for f in os.listdir(self.data_dir):
            if 'MR.nii' in f:
                if f in ['064MR.nii', '063MR.nii', '062MR.nii', '061MR.nii']:
                    test_mr_files.append(f)
                    test_ct_files.append(f[:3]+'CT.nii')
                else:
                    train_mr_files.append(f)
                    train_ct_files.append(f[:3]+'CT.nii')
        self._pre_data(train_mr_files, train_ct_files)
        self._pre_data(test_mr_files, test_ct_files, mode='test')
        self.train_datasets = self.datasets(self.data_dir)
        self.test_datasets = self.datasets(self.data_dir, 'test')
    
    def _pre_data(self, mr_files, ct_files, mode='train'):
        mode = mode.lower()
        assert mode in ['train', 'test', 'predict'], \
                "mode should be 'train' or 'test' or 'predict', but got {}".format(mode)
        if mode == 'train':
            mr_img_dir = 'trainMR'
            ct_img_dir = 'trainCT'
            img_list = os.path.join(self.data_dir, 'train.txt')
        elif mode == 'test':
            mr_img_dir = 'testMR'
            ct_img_dir = 'testCT'
            img_list = os.path.join(self.data_dir, 'test.txt')
        f = open(img_list, 'w')
        idx = 0
        for mr_f, ct_f in zip(mr_files, ct_files):
            mr_f, ct_f = os.path.join(self.data_dir, mr_f), os.path.join(self.data_dir, ct_f)
            mr_img = sitk.ReadImage(mr_f)
            ct_img = sitk.ReadImage(ct_f)
            mr_img = sitk.GetArrayFromImage(mr_img)
            ct_img = sitk.GetArrayFromImage(ct_img)
            # mr_img, ct_img = nib.load(mr_f), nib.load(ct_f)
            # mr_img, ct_img = np.asarray(mr_img.dataobj, dtype='float32'), np.asarray(ct_img.dataobj, 'float32')
            # ct_img = rm_nan_ct(ct_img)
            # ct_img = rm_neg(ct_img)
            # mr_img, ct_img = rm_max(mr_img), rm_max(ct_img)
            mr_img = mr_img / 255
            ct_img = ct_img / 255

            for mr_sub_img, ct_sub_img in zip(
                extract_ordered_overlap(mr_img, self.patch_shape, self.stride_shape),
                extract_ordered_overlap(ct_img, self.patch_shape, self.stride_shape)
                ):
                mr_name = "{}/mr{}".format(mr_img_dir, idx)
                ct_name = "{}/ct{}".format(ct_img_dir, idx)
                f.write(mr_name + '\t' + ct_name + '\n')
                mr_name = os.path.join(self.data_dir, mr_name)
                ct_name = os.path.join(self.data_dir, ct_name)
                np.save(mr_name, mr_sub_img.astype('float32'))
                np.save(ct_name, ct_sub_img.astype('float32'))
                idx += 1

    def train(self, epochs=1, batch_size=1, step_num=50, print_interval=1, load_model=False):
        pass
    
    def predict(self):
        model = paddle.Model(self.network)
        model.prepare(
            self._optimizer(model)
            
        )
        model.load(self.model_dir + self.model_name)
        preds = model.predict(self.test_datasets, batch_size=1)
        return preds
        

    def evaluate(self):
        preds = self.predict()
        out = np.array(preds)
        out = out[0][:,0]
        final_img = recompone_overlap(out, self.img_shape, self.stride_shape)
        # 合成真实图像
        text_list = os.path.join(self.data_dir, 'test.txt')
        ct_file_list = []
        with open(text_list, 'r') as f:
            line = f.readline()
            while line:
                line = line.strip().split('\t')
                ct_file_list.append(line[1])
                line = f.readline()
        gts = []
        for f in ct_file_list:
            file_name = os.path.join(self.data_dir, "{}.npy".format(f))
            d = np.load(file_name)
            gts.append(d[np.newaxis, :])
        gts = np.asarray(gts)
        gt_image = recompone_overlap(gts, self.img_shape, self.stride_shape)
        psnr = self._psnr(final_img[0, 0], gt_image[0, 0], 1)
        ssim = self._ssim(final_img[0, 0], gt_image[0, 0], L=1)
        print("PSNR: {}, SSIM: {}".format(psnr, ssim))
        return psnr, ssim, final_img, gt_image

    def _optimizer(self, model):
        return paddle.optimizer.Adam(learning_rate=0.0002, parameters=model.parameters())

    def _loss(self):
        return F.mse;

    def _psnr(self, img1, img2, max_pixel=1):
        img1, img2 = np.array(img1, dtype='float'), np.array(img2, dtype='float')
        mse = np.mean((img1 - img2) ** 2)
        psnr = 10 * np.log10(max_pixel * max_pixel / mse)
        return psnr

    # SSIM
    def _ssim(self, img1, img2, K=(0.01, 0.03), L=1):
        C1, C2 = (K[0] * L) ** 2, (K[1] * L) ** 2
        C3 = C2 / 2
        img1, img2 = np.array(img1, dtype='float'), np.array(img2, dtype='float')
        m1, m2 = np.mean(img1), np.mean(img2)
        s1, s2 = np.std(img1), np.std(img2)
        s12 = np.mean((img1 - m1) * (img2 - m2))
        l = (2 * m1 * m2 + C1) / (m1**2 + m2**2 + C1)
        c = (2 * s1 * s2 + C2) / (s1**2 + s2**2 + C2)
        s = (s12 + C3) / (s1 * s2 + C3)
        return l * c * s