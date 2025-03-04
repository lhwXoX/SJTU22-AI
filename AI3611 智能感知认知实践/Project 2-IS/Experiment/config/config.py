from easydict import EasyDict

cfg = EasyDict()

# model
cfg.d_input = 784
cfg.d_hidden = 256
cfg.d_latent = 64

# pretrain
cfg.pretrain = False
cfg.path_checkpoint = 'output/checkpoint/model.pkl'

# train
cfg.max_epochs = 200
cfg.batch_size = 128
cfg.learning_rate = 2e-3
cfg.epsilon = 1 # weight of regularization loss
cfg.save_interval = 20