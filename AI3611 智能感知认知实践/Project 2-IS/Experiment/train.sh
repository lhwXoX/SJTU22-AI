# epsilon
python main.py -save_name 'latent_64_epsilon_0.5' -epsilon 0.5
python main.py -save_name 'latent_64_epsilon_1.0' -epsilon 1.0
python main.py -save_name 'latent_64_epsilon_2.0' -epsilon 2.0

# latent dimension
python main.py -save_name 'latent_1_epsilon_1.0' -d_latent 1
python main.py -save_name 'latent_2_epsilon_1.0' -d_latent 2
python main.py -save_name 'latent_16_epsilon_1.0' -d_latent 16
python main.py -save_name 'latent_32_epsilon_1.0' -d_latent 32
python main.py -save_name 'latent_64_epsilon_1.0' -d_latent 64