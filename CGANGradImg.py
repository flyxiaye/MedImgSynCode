import paddle.nn.functional as F
import paddle
import numpy as np
import os
import SimpleITK as sitk
from myutiles import *
from visualdl import LogWriter
from itertools import product

from CGANTrainer import CGANTrainer

class CGANGradImg(CGANTrainer):

    def __init__(self, network, dsc_network, data_dir, model_dir, model_name, dsc_model_name, log_dir, datasets):
        super().__init__(network, dsc_network, data_dir, model_dir, model_name, dsc_model_name, log_dir, datasets)
        self.dsc_network = dsc_network(3)

    def train(self, epochs=1, batch_size=1, print_interval=100, save_interval=100, load_model=False):
        writer = LogWriter(logdir=self.log_dir+"train")
        # 加载训练集 batch_size 设为 1
        train_loader = paddle.io.DataLoader(self.train_datasets, batch_size=batch_size, shuffle=True)
        # 设置网络和优化器
        g_opt = self._optimizer(self.network)
        d_opt = self._dsc_optimizer(self.dsc_network)
        gen_loss_fun = self._loss()
        dsc_loss_fun = self._dsc_loss()
        
        # 继续训练
        if load_model:
            gen_state_dict = paddle.load(self.model_dir + "{}.pdparams".format(self.model_name))
            self.network.set_state_dict(gen_state_dict)
            dsc_state_dict = paddle.load(self.model_dir + "{}.pdparams".format(self.dsc_model_name))
            self.dsc_network.set_state_dict(dsc_state_dict)

        ones, zeros = None, None
        first_iter = True
        step = 0
        for epoch in range(epochs):
            for batch_id, data in enumerate(train_loader()):
                x_data = data[0]
                y_data = data[1]
                grad = data[2]
                predicts = self.network(x_data)
                pre_grads = np.zeros(shape=predicts.shape, dtype='float32')
                for i, j in product(range(predicts.shape[0]), range(predicts.shape[1])):
                    pre_grads[i, j] = sitk.GetArrayFromImage(
                        sitk.SobelEdgeDetection(
                            sitk.GetImageFromArray(predicts.numpy()[i, j])
                        )
                    )
                pre_grads = paddle.to_tensor(pre_grads)
                real_data = paddle.concat((x_data, y_data, grad), axis=1)
                fake_data = paddle.concat((x_data, predicts, pre_grads), axis=1)
                d_real = self.dsc_network(real_data)
                d_fake = self.dsc_network(fake_data)
                if first_iter:
                    first_iter = False
                    ones = paddle.ones_like(d_real)
                    zeros = paddle.zeros_like(d_fake)
                d_loss = dsc_loss_fun(dsc_real=d_real, dsc_fake=d_fake, zeros=zeros, ones=ones)
                d_loss.backward()
                d_opt.step()
                d_opt.clear_grad()
                
                x_data = data[0]
                predicts = self.network(x_data)
                pre_grads = np.zeros(shape=predicts.shape, dtype='float32')
                for i, j in product(range(predicts.shape[0]), range(predicts.shape[1])):
                    pre_grads[i, j] = sitk.GetArrayFromImage(
                        sitk.SobelEdgeDetection(
                            sitk.GetImageFromArray(predicts.numpy()[i, j])
                        )
                    )
                pre_grads = paddle.to_tensor(pre_grads)
                fake_data = paddle.concat((x_data, predicts, pre_grads), axis=1)
                g_fake = self.dsc_network(fake_data)
                g_loss = gen_loss_fun(dsc_fake=g_fake, zeros=zeros, predicts=predicts, y_data=y_data)
                g_loss.backward()
                g_opt.step()
                g_opt.clear_grad()

                writer.add_scalar(tag='g_loss', step=step, value=g_loss)
                writer.add_scalar(tag='d_loss', step=step, value=d_loss)
                step += 1
                
                if batch_id % print_interval == 0:
                    # print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
                    print("epoch: {}, batch_id: {}, g_loss is: {}, d_loss is: {}, acc is: {}".format(epoch, batch_id, g_loss.numpy(), d_loss.numpy(), 0))
                if (batch_id + 1) % save_interval == 0:
                    paddle.save(self.network.state_dict(), self.model_dir + "{}.pdparams".format(self.model_name))
                    paddle.save(self.dsc_network.state_dict(), self.model_dir + "{}.pdparams".format(self.dsc_model_name))
            # 每一轮结束，保存模型
            paddle.save(self.network.state_dict(), self.model_dir + "{}.pdparams".format(self.model_name))
            paddle.save(self.dsc_network.state_dict(), self.model_dir + "{}.pdparams".format(self.dsc_model_name))
        writer.close()

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
           
            ct_img = rm_nan_ct(ct_img)
            ct_img = rm_neg(ct_img)
            mr_img, ct_img = rm_max(mr_img), rm_max(ct_img)
            mr_img = mr_img / np.max(mr_img)
            ct_img = ct_img / np.max(ct_img)
            grad_img = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(sitk.GetImageFromArray(ct_img)))
            for mr_sub_img, ct_sub_img, grad_sub_img in zip(
                extract_ordered_overlap(mr_img, self.patch_shape, self.stride_shape),
                extract_ordered_overlap(ct_img, self.patch_shape, self.stride_shape),
                extract_ordered_overlap(grad_img, self.patch_shape, self.stride_shape),
                ):
                mr_name = "{}/mr{}".format(mr_img_dir, idx)
                ct_name = "{}/ct{}".format(ct_img_dir, idx)
                grad_name = "{}/grad{}".format(ct_img_dir, idx)
                f.write(mr_name + '\t' + ct_name + '\t' + grad_name + '\n')
                mr_name = os.path.join(self.data_dir, mr_name)
                ct_name = os.path.join(self.data_dir, ct_name)
                grad_name = os.path.join(self.data_dir, grad_name)
                np.save(mr_name, mr_sub_img.astype('float32'))
                np.save(ct_name, ct_sub_img.astype('float32'))
                np.save(grad_name, grad_sub_img.astype('float32'))
                idx += 1
