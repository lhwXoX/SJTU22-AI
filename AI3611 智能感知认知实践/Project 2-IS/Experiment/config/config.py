from easydict import EasyDict

cfg = EasyDict()

# model
cfg.d_input = 784
cfg.d_hidden = 256
cfg.d_latent = 64

# pretrain
cfg.pretrain = False
cfg.path_checkpoint = 'checkpoint/model.pkl'

# train
cfg.max_epochs = 200
cfg.batch_size = 128
cfg.learning_rate = 2e-3

# scheduler
cfg.factor = 0.5
cfg.patience = 5
cfg.threshold = 1e-3
cfg.min_lr = 1e-4