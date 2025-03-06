# interactive 3D latent distribution
python 3d_visualization.py -d_latent 16 -path_checkpoint 'output/latent_16_epsilon_1.0/checkpoint/checkpoint_epoch_250.pkl'
python 3d_visualization.py -d_latent 32 -path_checkpoint 'output/latent_32_epsilon_1.0/checkpoint/checkpoint_epoch_250.pkl'
python 3d_visualization.py -d_latent 64 -path_checkpoint 'output/latent_64_epsilon_1.0/checkpoint/checkpoint_epoch_250.pkl'