import os
import copy
import torch

class Callback:
    """
    Abstract class for callbacks.
    """

    def __init__(self):
        pass

    def __call__(self):
        pass

    def reset(self):
        pass

    def final(self, **kwargs):
        self.reset()

class ModelCheckpoint:
    """
    # TODO

    Arguments:
        path:
        num_iters: number of iterations after which to store the model.
            If set to -1, it will only store the last iteration's model.
        preped: string to prepend the filename with.
    """

    def __init__(self, path, prepend="", num_iters=-1, ignore_before=0):
        assert(os.path.isdir(path))
        self.path = path
        # end the prepended text with an underscore if it does not
        if not prepend.endswith("_"):
            prepend += "_"
        self.prepend = prepend
        self.num_iters = num_iters
        self.ignore_before = ignore_before

    def __call__(self, trainer, epoch, val_metrics):
        if not self.num_iters == -1:
            if epoch >= self.ignore_before:
                if epoch % self.num_iters == 0:
                    # TODO: prevent overwriting in cross validation!!
                    name = self.prepend + "training_epoch_{}.h5".format(epoch)
                    full_path = os.path.join(self.path, name)
                    self.save_model(trainer, full_path)

    def reset(self):
        """
        Reset module after training.
        Useful for cross validation.
        """

    def final(self, **kwargs):
        epoch = kwargs["epoch"]
        if epoch >= self.ignore_before:
            name = self.prepend + "training_epoch_{}_FINAL.h5".format(epoch)
            full_path = os.path.join(self.path, name)
            self.save_model(kwargs["trainer"], full_path)
        else:
            print("Minimum iterations to store model not reached.")
            self.reset()
            return self.best_res, best_model

    def save_model(self, trainer, full_path):
        model = trainer.model.cpu()
        torch.save(model.state_dict(), full_path)
        if trainer.gpu is not None:
            trainer.model.cuda(trainer.gpu)


class EarlyStopping:
    """ 
    Stop training when a monitored quantity has stopped improving.

    Arguments
        patience: number of iterations without improvement after which
            to stop
        retain_metric: the metric which you want to monitor
        mode: {min or max}; defines if you want to maximise or minimise
            your metric
        ignore_before: does not start the first window until this epoch.
            Can be useful when training spikes a lot in early epochs.
    """


    def __init__(self, patience, retain_metric, mode, ignore_before=0):
        self.patience = patience
        self.retain_metric = retain_metric
        self.mode = mode
        self.ignore_before = ignore_before
        self.best_res = -1
        # set to first iteration which is interesting
        self.best_epoch = self.ignore_before

    def __call__(self, trainer, epoch, val_metrics):
        if epoch >= self.ignore_before:
            if epoch - self.best_epoch < self.patience:
                if isinstance(self.retain_metric, str):
                    current_res = val_metrics[self.retain_metric][-1]
                else:
                    current_res = val_metrics[self.retain_metric.__name__][-1]
                if self.compare(current_res):
                    self.best_res = current_res
                    self.best_epoch = epoch
            else:
                # end training run
                trainer.stop_training = True

    def compare(self, res):
        if self.mode == "max":
            return res > self.best_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_res == -1:
                return True
            else:
                return res < self.best_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def reset(self):
        """ Resets after training. Useful for cross validation."""
        self.best_res = -1
        self.best_epoch = self.ignore_before

    def final(self, **kwargs):
        self.reset()