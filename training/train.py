from utils.seed import set_seed
from training.trainer import train

if __name__ == "__main__":
    set_seed(42)
    train()