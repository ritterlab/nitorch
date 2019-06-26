import os
from copy import deepcopy
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# pytorch
import torch
import torch.nn as nn

# nitorch
from nitorch.data import show_brain


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


class ModelCheckpoint(Callback):
    """
    # TODO

    Arguments:
        path:
        num_iters: number of iterations after which to store the model.
            If set to -1, it will only store the last iteration's model.
        prepend: string to prepend the filename with.
        ignore_before: ignore early iterations.
        store_best: boolen whether to save the best model during
            training.
        store_best_metric: name of the metric to use for best model
            selection.
        mode: "max" or "min".
    """

    def __init__(
            self,
            path,
            retain_metric="loss",
            prepend="",
            num_iters=-1,
            ignore_before=0,
            store_best=False,
            mode="max",
            window=None,
            info=False
    ):
        super().__init__()
        if os.path.isdir(path):
            self.path = path
        else:
            os.makedirs(path)
            self.path = path
        # end the prepended text with an underscore if it does not
        if not prepend.endswith("_") and prepend != "":
            prepend += "_"
        self.prepend = prepend
        self.num_iters = num_iters
        self.ignore_before = ignore_before
        self.best_model = None
        self.best_res = -1
        self.best_mean_res = -1
        self.best_window_start = -1
        self.current_window_best_res = -1
        self.current_window_best_epoch = -1
        self.current_window_save_idx = -1
        self.current_window_best_model_save_idx = 0
        if window:
            self.state_dict_storage = [0] * window
        else:
            self.state_dict_storage = [0]
        self.store_best = store_best
        self.retain_metric = retain_metric
        self.mode = mode
        self.window = window
        self.info = info

    def __call__(self, trainer, epoch):
        # do not store intermediate iterations
        if epoch >= self.ignore_before and epoch != 0:
            if not self.num_iters == -1:

                # counting epochs starts from 1; i.e. +1
                epoch += 1
                # store model recurrently if set
                if epoch % self.num_iters == 0:
                    name = self.prepend + "training_epoch_{}.h5".format(epoch)
                    full_path = os.path.join(self.path, name)
                    self.save_model(trainer, full_path)

            # store current model if improvement detected
            if self.store_best:
                current_res = 0
                try:
                    # check if value can be used directly or not
                    if isinstance(self.retain_metric, str):
                        current_res = trainer.val_metrics[self.retain_metric][-1]
                    else:
                        current_res = trainer.val_metrics[self.retain_metric.__name__][-1]
                except KeyError:
                    print("Couldn't find {} in validation metrics. Using \
                        loss instead.".format(self.retain_metric))
                    current_res = trainer.val_metrics["loss"][-1]

                # update
                if self.window is None:  # old update style
                    if self.has_improved(current_res):
                        self.best_res = current_res
                        self.best_model = deepcopy(trainer.model.state_dict())
                else:  # new update style
                    # get validation metrics in certain window
                    try:
                        if isinstance(self.retain_metric, str):
                            start = len(trainer.val_metrics[self.retain_metric]) - self.window
                            start = 0 if start < 0 else start

                            window_val_metrics = trainer.val_metrics[self.retain_metric][start:]
                        else:
                            start = len(trainer.val_metrics[self.retain_metric.__name__]) - self.window
                            start = 0 if start < 0 else start
                            window_val_metrics = trainer.val_metrics[self.retain_metric.__name__][start:]
                    except KeyError:
                        print(
                            "Couldn't find {} in validation metrics. Using \
                            loss instead.".format(
                                self.retain_metric
                            )
                        )
                        start = len(trainer.val_metrics[self.retain_metric]) - self.window
                        start = 0 if start < 0 else start
                        window_val_metrics = trainer.val_metrics["loss"][start:]

                    # build mean
                    mean_window_res = np.mean(window_val_metrics)

                    # only safe when improvement to previous epoch detected
                    # only a value BETTER than before can be the minimum/maximum of a
                    # window with better mean than a previously detected window
                    if len(window_val_metrics) == 1 \
                            or self.first_val_better(window_val_metrics[-1], window_val_metrics[-2]) \
                            or self.current_window_save_idx == -1:
                        if self.current_window_save_idx == -1:
                            self.current_window_save_idx = 0
                        self.state_dict_storage[self.current_window_save_idx] = deepcopy(trainer.model.state_dict())
                        # increase save idx and take modulo
                        self.current_window_save_idx += 1
                        self.current_window_save_idx = divmod(self.current_window_save_idx, self.window)[1]
                    else:  # only increase current_window_save_idx (for modulo index calculation to work)
                        self.current_window_save_idx += 1
                        self.current_window_save_idx = divmod(self.current_window_save_idx, self.window)[1]

                    # always update current window best result - it might be at some point overall best result
                    current_window_best_idx = self.get_cur_win_best_idx(window_val_metrics)
                    if current_window_best_idx == len(window_val_metrics) - 1 \
                            or len(window_val_metrics) == 1:  # case of improvement or initialisation
                        # overwrite model_state saved so far
                        self.current_window_best_model_save_idx = self.current_window_save_idx
                        self.current_window_best_epoch = epoch
                        self.current_window_best_res = window_val_metrics[-1]

                    # check if mean has improved and copy values as best model result
                    if self.has_window_mean_improved(mean_window_res):
                        self.best_mean_res = mean_window_res
                        self.best_window_start = 0 if epoch - self.window + 1 < 0 else epoch - self.window + 1
                        # save current window best as overall best
                        self.best_res = self.current_window_best_res
                        self.best_model = copy.deepcopy(self.state_dict_storage[self.current_window_best_model_save_idx])
                        if self.info:
                            print("Found a window with better validation metric mean:")
                            print("\t metric mean: {}".format(mean_window_res))
                            print("\t epoch start: {}".format(self.best_window_start))
                            print("\t best result: {}".format(self.best_res))

    def first_val_better(self, v1, v2):
        if self.mode == "max":
            return v1 >= v2
        elif self.mode == "min":
            return v1 <= v2
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def get_cur_win_best_idx(self, val_metr):
        if self.mode == "max":
            return val_metr.index(max(val_metr))
        elif self.mode == "min":
            return val_metr.index(min(val_metr))
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def has_window_mean_improved(self, mean_window_res):
        if self.mode == "max":
            return mean_window_res >= self.best_mean_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_mean_res == -1:
                return True
            else:
                return mean_window_res <= self.best_mean_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def reset(self):
        """
        Reset module after training.
        Useful for cross validation.
        """
        self.best_model = None
        self.best_res = -1

    def final(self, **kwargs):
        epoch = kwargs["epoch"] + 1
        if epoch >= self.ignore_before:
            name = self.prepend + "training_epoch_{}_FINAL.h5".format(epoch)
            full_path = os.path.join(self.path, name)
            self.save_model(kwargs["trainer"], full_path)
        else:
            print("Minimum iterations to store model not reached.")

        if self.best_model is not None:
            best_model = deepcopy(self.best_model)
            best_res = self.best_res
            if self.window is not None:
                print("Best result during training: {:.2f}.\n In a window of size {} "
                      "starting in epoch {} with best mean value of {} \n Saving model..".format(
                    best_res, self.window, self.best_window_start, self.best_mean_res
                ))
            else:
                print(
                    "Best result during training: {:.2f}. Saving model..".format(
                        best_res
                    )
                )
            name = self.prepend + "BEST_ITERATION.h5"
            torch.save(best_model, os.path.join(self.path, name))
        self.reset()

    @staticmethod
    def save_model(trainer, full_path):
        print("Writing model to disk...")
        model = trainer.model.cpu()
        torch.save(model.state_dict(), full_path)
        if trainer.device is not None:
            trainer.model.cuda(trainer.device)

    def has_improved(self, res):
        if self.mode == "max":
            return res >= self.best_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_res == -1:
                return True
            else:
                return res <= self.best_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")


class EarlyStopping(Callback):
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

    def __init__(self, patience, retain_metric, mode, ignore_before=0, window=None, info=False):
        super().__init__()
        self.patience = patience
        self.retain_metric = retain_metric
        self.mode = mode
        self.ignore_before = ignore_before
        self.best_res = -1
        # set to first iteration which is interesting
        self.best_epoch = self.ignore_before
        # window mod
        self.best_mean_res = -1
        self.best_window_start = -1
        self.current_window_best_res = -1
        self.current_window_best_epoch = -1
        self.current_window_save_idx = -1
        self.current_window_best_model_save_idx = 0
        if window:
            self.state_dict_storage = [0] * window
        else:
            self.state_dict_storage = [0]
        self.mode = mode
        self.window = window
        self.info = info

    def first_val_better(self, v1, v2):
        if self.mode == "max":
            return v1 >= v2
        elif self.mode == "min":
            return v1 <= v2
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def get_cur_win_best_idx(self, val_metr):
        if self.mode == "max":
            return val_metr.index(max(val_metr))
        elif self.mode == "min":
            return val_metr.index(min(val_metr))
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def has_window_mean_improved(self, mean_window_res):
        if self.mode == "max":
            return mean_window_res >= self.best_mean_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_mean_res == -1:
                return True
            else:
                return mean_window_res <= self.best_mean_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def __call__(self, trainer, epoch):
        if epoch >= self.ignore_before:
            if epoch - self.best_epoch < self.patience:
                if isinstance(self.retain_metric, str):
                    current_res = trainer.val_metrics[self.retain_metric][-1]
                else:
                    current_res = trainer.val_metrics[self.retain_metric.__name__][-1]
                if self.window is None:
                    if self.has_improved(current_res):
                        self.best_epoch = epoch
                        self.best_res = current_res
                        trainer.best_metric = current_res
                        trainer.best_model = trainer.model
                else:  # window mod
                    # get validation metrics in certain window
                    try:
                        if isinstance(self.retain_metric, str):
                            start = len(trainer.val_metrics[self.retain_metric]) - self.window
                            start = 0 if start < 0 else start

                            window_val_metrics = trainer.val_metrics[self.retain_metric][start:]
                        else:
                            start = len(trainer.val_metrics[self.retain_metric.__name__]) - self.window
                            start = 0 if start < 0 else start
                            window_val_metrics = trainer.val_metrics[self.retain_metric.__name__][start:]
                    except KeyError:
                        print(
                            "Couldn't find {} in validation metrics. Using \
                            loss instead.".format(
                                self.retain_metric
                            )
                        )
                        start = len(trainer.val_metrics[self.retain_metric]) - self.window
                        start = 0 if start < 0 else start
                        window_val_metrics = trainer.val_metrics["loss"][start:]

                    # build mean
                    mean_window_res = np.mean(window_val_metrics)

                    # only safe when improvement to previous epoch detected
                    # only a value BETTER than before can be the minimum/maximum of a
                    # window with better mean than a previously detected window
                    if len(window_val_metrics) == 1 \
                            or self.first_val_better(window_val_metrics[-1], window_val_metrics[-2]) \
                            or self.current_window_save_idx == -1:
                        if self.current_window_save_idx == -1:
                            self.current_window_save_idx = 0
                        self.state_dict_storage[self.current_window_save_idx] = deepcopy(trainer.model.state_dict())
                        # increase save idx and take modulo
                        self.current_window_save_idx += 1
                        self.current_window_save_idx = divmod(self.current_window_save_idx, self.window)[1]
                    else:  # only increase current_window_save_idx (for modulo index calculation to work)
                        self.current_window_save_idx += 1
                        self.current_window_save_idx = divmod(self.current_window_save_idx, self.window)[1]

                    # always update current window best result - it might be at some point overall best result
                    current_window_best_idx = self.get_cur_win_best_idx(window_val_metrics)
                    if current_window_best_idx == len(window_val_metrics) - 1 \
                            or len(window_val_metrics) == 1:  # case of improvement or initialisation
                        # overwrite model_state saved so far
                        self.current_window_best_model_save_idx = self.current_window_save_idx
                        self.current_window_best_epoch = epoch
                        self.current_window_best_res = window_val_metrics[-1]

                    # check if mean has improved and copy values as best model result
                    if self.has_window_mean_improved(mean_window_res):
                        self.best_mean_res = mean_window_res
                        self.best_window_start = 0 if epoch - self.window + 1 < 0 else epoch - self.window + 1
                        # save current window best as overall best
                        self.best_res = self.current_window_best_res
                        self.best_model = copy.deepcopy(self.state_dict_storage[self.current_window_best_model_save_idx])
                        self.best_epoch = self.current_window_best_epoch
                        trainer.best_metric = self.current_window_best_res
                        trainer.best_model = trainer.model
                        if self.info:
                            print("Found a window with better validation metric mean:")
                            print("\t metric mean: {}".format(mean_window_res))
                            print("\t epoch start: {}".format(self.best_window_start))
                            print("\t best result: {}".format(self.best_res))

            else:
                # end training run
                trainer.stop_training = True
                if self.window is None:
                    print("Early stopping at epoch {}.\nBest model was at epoch {} with val metric score = {}".format(
                        epoch, self.best_epoch, self.best_res)
                    )
                else:
                    print("Early stopping with window mode at epoch {}.\n"
                          "Best results were achieved at epoch {} with val metric score = {}.\n"
                          "Best window of size {} achieved a mean result of {} and started at epoch {}.".format(
                        epoch, self.best_epoch, self.best_res, self.window, self.best_mean_res, self.best_window_start)
                    )

    def has_improved(self, res):
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


# Functions which can be used in custom-callbacks for visualizing 3D-features during training
# (using the argument 'training_time_callback' in nitorch's Trainer class )
def visualize_feature_maps(features, return_fig=False):
    if features.is_cuda:
        features = features.cpu().detach().numpy()

    num_features = len(features)
    plt.close('all')
    n = int(math.log2(num_features))
    fig_size = (n * 2, n * 6)
    fig = plt.figure(figsize=fig_size)

    for i, f in enumerate(features, 1):
        # normalize to range [0, 1] first as the values can be very small
        if (f.max() - f.min()) != 0:
            f = (f - f.min()) / (f.max() - f.min())

            idxs = np.nonzero(f)
            vals = np.ravel(f[idxs])
            if len(vals):
                # calculate the index where the mean value would lie
                mean_idx = np.average(idxs, axis=1, weights=vals)
                # calculate the angel ratios for each non-zero val
                angles = (mean_idx.reshape(-1, 1) - idxs)
                angles = angles / (np.max(abs(angles), axis=1).reshape(-1, 1))
            else:  # if all values in f are zero, set dummy angle
                angles = [1, 1, 1]

            #             print("values = ",vals)
            ax = fig.add_subplot(num_features // 3 + 1, 3, i,
                                 projection='3d')
            ax.set_title("Feature-{} in the bottleneck".format(i))
            ax.quiver(*idxs, angles[0] * vals, angles[1] * vals, angles[2] * vals)
            plt.grid()

        else:
            ax = fig.add_subplot(num_features // 3 + 1, 3, i)
            ax.text(0.5, 0.5, "All values zero!", transform=ax.transAxes)
            plt.axis('off')

    plt.tight_layout()
    if return_fig:
        return fig


class CAE_VisualizeTraining(Callback):
    """
    training_time_callback that prints the model dimensions,
    visualizes CAE encoder outputs, original image and reconstructed image
    during training.

    NOTE : The forward() function of the CAE model using this callback
    must return a (decoder_output, encoder_output) tuple.
    """

    def __init__(self, model, max_train_iters, show_epochs_list=[], plotFeatures=True, plot_pdf_path="",
                 cmap="nipy_spectral"):
        super().__init__()
        self.model = model
        self.max_train_iters = max_train_iters
        if plot_pdf_path is not None:
            assert isinstance(plot_pdf_path, str), "pp is not a path!"
        self.plot_pdf_path = plot_pdf_path
        assert isinstance(plotFeatures, bool), "plotFeatures not boolean object!"
        self.plotFeatures = plotFeatures
        assert isinstance(show_epochs_list, list), "show_epochs_list is not a list!"
        self.show_epochs_list = show_epochs_list
        self.cmap = cmap
        self.ave_grads = []
        self.layers = []
        # inform the model to also return the encoder output along with the decoder output
        try:
            if isinstance(model, nn.DataParallel):
                model.module.set_return_encoder_out(True)
            else:
                model.set_return_encoder_out(True)
        except AttributeError:
            raise "The CAE model must implement a setter function 'set_return_encoder_out'\
 for a flag 'encoder_out' which when set to true, the forward() function using this callback \
must return a (decoder_output, encoder_output) tuple instead of just (encoder_output). See the CAE class in models.py for the framework."

    def __call__(self, inputs, labels, train_iter, epoch):
        debug = False
        visualize_training = False
        tmp_show_epoches_list = []

        # if show_epochs_list is empty, all epoches should be plotted. Therefore, add current epoch to the list
        if not self.show_epochs_list:
            tmp_show_epoches_list.append(epoch)
        else:
            tmp_show_epoches_list = self.show_epochs_list

        # check if epoch should be visualized
        if epoch in tmp_show_epoches_list:
            # print the model's parameter dimensions etc in the first iter
            if train_iter == 0 and epoch == 0:
                debug = True
            # visualize training on the last iteration in that epoch
            elif (train_iter == 1 and epoch == 0) or (train_iter == self.max_train_iters):
                visualize_training = True

        # for nitorch models which have a 'debug' and 'visualize_training' switch in the
        # forward() method

        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_debug(debug)
        else:
            self.model.set_debug(debug)

        outputs, encoder_out = self.model(inputs)

        if visualize_training:
            # check if result should be plotted in PDF
            if self.plot_pdf_path != "":
                pp = PdfPages(os.path.join(self.plot_pdf_path, "training_epoch_" + str(epoch) + "_visualization.pdf"))
            else:
                pp = None

            # show only the first image in the batch
            if pp is None:
                # input image
                show_brain(inputs[0].squeeze().cpu().detach().numpy(), draw_cross=False, cmap=self.cmap)
                plt.suptitle("Input image")
                plt.show()
                if not torch.all(torch.eq(inputs[0], labels[0])):
                    show_brain(labels[0].squeeze().cpu().detach().numpy(), draw_cross=False, cmap=self.cmap)
                    plt.suptitle("Expected reconstruction")
                    plt.show()
                    # reconstructed image
                show_brain(outputs[0].squeeze().cpu().detach().numpy(), draw_cross=False, cmap=self.cmap)
                plt.suptitle("Reconstructed Image")
                plt.show()
                # statistics
                print(
                    "\nStatistics of expected reconstruction:\n(min, max)=({:.4f}, {:.4f})\nmean={:.4f}\nstd={:.4f}".format(
                        labels[0].min(), labels[0].max(), labels[0].mean(), labels[0].std()))
                print(
                    "\nStatistics of Reconstructed image:\n(min, max)=({:.4f}, {:.4f})\nmean={:.4f}\nstd={:.4f}".format(
                        outputs[0].min(), outputs[0].max(), outputs[0].mean(), outputs[0].std()))
                # feature maps
                visualize_feature_maps(encoder_out[0])
                plt.suptitle("Encoder output")
                plt.show()
            else:
                # input image
                fig = show_brain(inputs[0].squeeze().cpu().detach().numpy(), draw_cross=False, return_fig=True,
                                 cmap=self.cmap)
                plt.suptitle("Input image")
                pp.savefig(fig)
                plt.close(fig)
                if not torch.all(torch.eq(inputs[0], labels[0])):
                    fig = show_brain(labels[0].squeeze().cpu().detach().numpy(), draw_cross=False, cmap=self.cmap)
                    plt.suptitle("Expected reconstruction")
                    pp.savefig(fig)
                    plt.close(fig)
                # reconstructed image
                fig = show_brain(outputs[0].squeeze().cpu().detach().numpy(), draw_cross=False, return_fig=True,
                                 cmap=self.cmap)
                plt.suptitle("Reconstructed Image")
                pp.savefig(fig)
                plt.close(fig)
                # feature maps
                if self.plotFeatures:
                    fig = visualize_feature_maps(encoder_out[0], return_fig=True)
                    plt.suptitle("Encoder output")
                    pp.savefig(fig)
                    plt.close(fig)

            # close the PDF
            if pp is not None:
                pp.close()

        if isinstance(self.model, nn.DataParallel):
            self.model.module.set_debug(False)
        else:
            self.model.set_debug(False)

        return outputs
