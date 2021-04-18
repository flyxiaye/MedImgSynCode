from datasets.data_set import PetDataset
from datasets.gradImg import GradImg
import network.cgan

from CGANTrainer import CGANTrainer
from CGANGradLoss import CGANGradLoss
from CGANGradImg import CGANGradImg


def trainer_factory(train_name, data_dir='data/data75500'):
    train_name.lower()
    if train_name == 'cgan':
        return CGANTrainer(network.cgan.Gen, network.cgan.Dsc, data_dir, './model/', 'gen', 'dsc', './log/gan/', PetDataset)
    elif train_name == 'cgan_grad_loss':
        return CGANGradLoss(network.cgan.Gen, network.cgan.Dsc, data_dir, './model/gd_loss', 'gen', 'dsc', './log/gan_gd_loss/', PetDataset)
    elif train_name == 'cgan_grad_img':
        return CGANGradImg(network.cgan.Gen, network.cgan.Dsc, data_dir, './model/gd_img', 'gen', 'dsc', './log/gan_gd_img/', GradImg)
    else:
        print('No Trainer')
        return None