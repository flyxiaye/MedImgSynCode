from BaseTrainer import BaseTrainer
import paddle.nn.functional as F
from visualdl import LogWriter
import paddle

class CGANTrainer(BaseTrainer):

    def __init__(self, network, dsc_network, data_dir, model_dir, model_name, dsc_model_name, log_dir, datasets):
        super(CGANTrainer, self).__init__(network, data_dir, model_dir, model_name, log_dir, datasets)
        self.dsc_network = dsc_network()
        self.dsc_model_name = dsc_model_name

        self.lambda_l1 = 100

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
                predicts = self.network(x_data)
                real_data = paddle.concat((x_data, y_data), axis=1)
                fake_data = paddle.concat((x_data, predicts), axis=1)
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
                fake_data = paddle.concat((x_data, predicts), axis=1)
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

    def _optimizer(self, model):
        return paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999, parameters=model.parameters())

    def _dsc_optimizer(self, model):
        return paddle.optimizer.Adam(learning_rate=0.0002, beta1=0.5, beta2=0.999, parameters=model.parameters())

    def _loss(self):
        def gen_loss(dsc_fake=None, dsc_real=None, predicts=None, y_data=None, zeros=None, ones=None):
            dsc_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
            g_loss_l1 = F.l1_loss(y_data, predicts)
            return dsc_fake_loss + g_loss_l1 * self.lambda_l1
        return gen_loss

    def _dsc_loss(self):
        def dsc_loss(dsc_fake=None, dsc_real=None, predicts=None, y_data=None, zeros=None, ones=None):
            d_real_loss = F.binary_cross_entropy_with_logits(dsc_real, ones)
            d_fake_loss = F.binary_cross_entropy_with_logits(dsc_fake, zeros)
            return d_real_loss + d_fake_loss
        return dsc_loss
