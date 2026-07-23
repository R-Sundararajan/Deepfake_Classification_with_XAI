class EarlyStopping:
    """
    Stops training when validation accuracy
    does not improve for 'patience' epochs.
    """

    def __init__(self, patience=5):

        self.patience = patience

        self.counter = 0

        self.best_score = None

        self.early_stop = False

    def __call__(self, validation_accuracy):

        if self.best_score is None:

            self.best_score = validation_accuracy

            return

        if validation_accuracy > self.best_score:

            self.best_score = validation_accuracy

            self.counter = 0

        else:

            self.counter += 1

            print(
                f"No improvement "
                f"({self.counter}/{self.patience})"
            )

            if self.counter >= self.patience:

                self.early_stop = True