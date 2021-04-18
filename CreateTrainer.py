from datasets.data_set import PetDataset
from network.cgan import Gen
from network.cgan import Dsc

from CGANTrainer import CGANTrainer


def trainer_factory(train_name):
    train_name.lower()
    if train_name == 'cgan':
        return CGANTrainer(Gen, Dsc, 'data/data75500', './model/', 'gen', 'dsc', './log/gan/', PetDataset)
    elif train_name == 'cgan_grad_loss':
        pass
    elif train_name == 'cgan_grad_img':
        pass
    else:
        print('No Trainer')
        return None