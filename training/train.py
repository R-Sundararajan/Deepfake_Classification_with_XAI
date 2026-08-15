from utils.seed import set_seed
from training.trainer import train


def main(resume: bool = None, max_epochs: int = None):
    set_seed(42)
    # train(resume=resume, max_epochs=max_epochs)
    train(resume=resume, max_epochs=max_epochs)


if __name__ == "__main__":
    main()

