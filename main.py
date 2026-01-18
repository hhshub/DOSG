from mdl1 import model1
from config import Config

def elia_wind(mode, config):
    gan = model1(config)

    if mode == 0:  # Restart training
        gan.train(epoch=50)
    elif mode == 1:  # Load existing model and continue training
        gan.loadWeights()
        gan.train(epoch=20)
    # elif mode == 2:  # Generate random samples
    #     gan.loadWeights()
    #     gan.generate()
    else:
        return

if __name__ == "__main__":
    config = Config()
    elia_wind(0, config)