import paddle.nn.functional as F
import paddle
import numpy as np

from CGANTrainer import CGANTrainer


class CGANGradLoss(CGANTrainer):

    def __init__(self, network, dsc_network, data_dir, model_dir, model_name, dsc_model_name, log_dir, datasets):
        super().__init__(network, dsc_network, data_dir, model_dir, model_name, dsc_model_name, log_dir, datasets)
    
    def _loss(self):
        def gen_loss(dsc_fake=None, dsc_real=None, predicts=None, y_data=None, zeros=None, ones=None):
            dsc_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
            g_loss_l1 = F.l1_loss(y_data, predicts)
            input = y_data.numpy()
            label = predicts.numpy()
            grad_x = F.mse_loss(paddle.to_tensor(np.abs(np.gradient(input, axis=2))), paddle.to_tensor(np.abs(np.gradient(label, axis=2))))
            grad_y = F.mse_loss(paddle.to_tensor(np.abs(np.gradient(input, axis=3))), paddle.to_tensor(np.abs(np.gradient(label, axis=3))))
            grad_z = F.mse_loss(paddle.to_tensor(np.abs(np.gradient(input, axis=4))), paddle.to_tensor(np.abs(np.gradient(label, axis=4))))
            g_loss_gd = grad_x + grad_y + grad_z
            return dsc_fake_loss + g_loss_l1 * self.lambda_l1 + g_loss_gd
        return gen_loss

