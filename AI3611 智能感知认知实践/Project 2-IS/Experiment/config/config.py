import argparse
from datetime import datetime

def parse_args():
    parse = argparse.ArgumentParser()
    # model
    parse.add_argument('-d_input', type=int, default=784, help='input dimension, 28 x 28 for MNIST')
    parse.add_argument('-d_hidden', type=int, default=256, help='hidden dimension of VAE')
    parse.add_argument('-d_latent', type=int, default=64, help='latent dimension of VAE')
    parse.add_argument('-save_name', type=str, default=datetime.strftime(datetime.now(), '%m-%d_%H-%M'), help='the name of the folder to save model and figure')
    
    # pretrain
    parse.add_argument('-pretrain', type=bool, default=False, help='load pretrained model')
    parse.add_argument('-path_checkpoint', type=str, default='output/xxx/checkpoint/checkpoint_best.pkl', help='the path to checkpoint to load')
    
    # training
    parse.add_argument('-max_epochs', type=int, default=1, help='the number of largest epoch to train')
    parse.add_argument('-batch_size', type=int, default=256, help='batch size for dataloader')
    parse.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parse.add_argument('-epsilon', type=float, default=1.0, help='weight of regularization loss')
    parse.add_argument('-save_interval', type=int, default=100, help='save checkpoint every certain epochs')
    
    opt = parse.parse_args()
    return opt