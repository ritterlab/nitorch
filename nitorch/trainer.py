from os.path import join
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_value_
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from nitorch.inference import predict
from nitorch.utils import *
import json


class Trainer:
    """Class for organizing the training process.

    Parameters
    ----------
    model
        Neural network to train.
    criterion
        The loss function.
    optimizer
        optimizer function.
    scheduler
        schedules the optimizer. Default: None
    metrics : list
        list of metrics to report. Default: None.
        when multitask training = True,
        metrics can be a list of lists such that len(metrics) =  number of tasks.
        If not, metrics are calculated only for the first task.
    callbacks
        list of callbacks to execute at the end of training epochs. Default: None.
    training_time_callback
        a user-defined callback that executes the model.forward() and returns the output to the trainer.
        This can be used to perform debug during train time, Visualize features,
        call model.forward() with custom arguments, run multiple decoder networks etc. Default: None.
    device : int/torch.device
        The device to use for training. Must be integer or a torch.device object.
        By default, GPU with current node is used. Default: torch.device("cuda")
    task_type : str
        accepts one of ["classif_binary", "classif", "regression", "other"].
        Default: "classif_binary"
    multitask : bool
        Enables multitask training. Default: False
    kwargs
        Other parameters to store.

    Useful Attributes
    ----------
    val_metrics : dict
        Lists as many metrics as specified in 'metrics' for each validation epoch. Always has "loss" as entry.
    train_metrics : dict
        Lists as many metrics as specified in 'metrics' for each training epoch. Always has "loss" as entry.
    best_metric
        Best validation metric.
    best_model
        Best model (hyperparameter settings) when 'best_metric' is archieved.
    start_time
        Time training started.

    Methods
    -------
    train_model()
        Main function to train a network for one epoch.
    finish_training()
        Function which should always be run when training ends.
    visualize_training()
        Function to visualize training process
    evaluate_model()
        Function to evaluate a model once it is trained

    """
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler=None,
            metrics=[],
            callbacks=[],
            training_time_callback=None,
            device=torch.device("cuda"),
            task_type="classif_binary",
            multitask=False,
            **kwargs
    ):
        """Initialization routine.

        Raises
        ------
        ValueError
            Wrong device selected.
            'model' in wrong format.

        """
        if not isinstance(model, nn.Module):
            raise ValueError("Expects model type to be torch.nn.Module")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.multitask = multitask
        if self.multitask:
            self.metrics = metrics
            self.task_type = task_type
            self._criterions = criterion.loss_function
        else:
            self.metrics = [metrics]
            self.task_type = [task_type]
            self._criterions = [criterion]

        self.callbacks = callbacks
        self.training_time_callback = training_time_callback

        if isinstance(device, int):
            self.device = torch.device("cuda:" + str(device))
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError("Device needs to be of type torch.device or \
                integer.")
        self._stop_training = False
        self.start_time = None
        self.val_metrics = {"loss": []}
        self.train_metrics = {"loss": []}
        self.best_metric = None
        self.best_model = None
        self.kwargs = kwargs
        
    def arrange_data(
            self,
            data,
            inputs_key,
            labels_key
    ):
        """Extracts the inputs and labels from the data loader and moves them to the 
        analysis device. In case of multiple inputs or multiple outputs uses a list.

        In case the DataLoader does not output a named dictionary, the features
        are expected at index 0 and labels and index 1.

        Attributes
        ----------
        data : torch.utils.DataLoader
            DataLoader for the current set e.g. train, val, test.
        inputs_key : str
            In case the DataLoader outputs a named pair use this key for the 
            features.
        labels_key : str
            In case the DataLoader outputs a named pair use this key for the 
            labels.

        Returns
        -------
        inputs
            torch.Tensor of all features or list of torch.Tensors
        labels
            torch.Tensor of all labels or list of torch.Tensors
        """
        try:
            if isinstance(data, dict):
                inputs, labels = data[inputs_key], data[labels_key]
            else:
                # if data does not come in dictionary, assume
                # that data is ordered like [input, label]
                inputs, labels = data[0], data[1]
        except TypeError:
            raise TypeError("Data returned from the dataloaders \
              is not in the expected format.")

        # in case of multi-input or output create a list
        if isinstance(inputs, list):
            inputs = [inp.to(self.device) for inp in inputs]
        else:
            inputs = inputs.to(self.device)
        if isinstance(labels, list):
            labels = [label.to(self.device) for label in labels]
        else:
            labels = labels.to(self.device)
        return inputs, labels

    def train_model(
            self,
            train_loader,
            val_loader,
            inputs_key="image",
            labels_key="label",
            num_epochs=25,
            show_train_steps=None,
            show_validation_epochs=1,
            show_grad_flow=False
    ):
        """Main function to train a network for one epoch.

        Parameters
        ----------
        train_loader : torch.utils.DataLoader
            A pytorch Dataset iterator for training data.
        val_loader : torch.utils.DataLoader
            A pytorch Dataset iterator for validation data.
        inputs_key, labels_key
            The data returned by 'train_loader' and 'val_loader' can either be a dict of format
            data_loader[X_key] = inputs and data_loader[y_key] = labels
            or a list with data_loader[0] = inputs and data_loader[1] = labels.
            The default keys are "image" and "label".
        num_epochs
            The maximum number of epochs. Default: 25
        show_train_steps
            The number of training steps to show. Default: None
        show_validation_epochs
            Specifies every 'x' validation epoch to show. If set to 1 all epochs are shown. Default: 1
        show_grad_flow
            Visualize the gradient flow through the model during training. 
            If a path is given the gradient flow plot is saved at that path instead. Default: False.

        Returns
        -------
        tuple
            First entry is the trained model, second entry is a dictionary containing information on training procedure.

        See Also
        --------
        finish_training(epoch)

        Raises
        ------
        AssertionError
            If 'show_train_steps' smaller 0 or greater than the length of the train loader.
        TypeError
            When data cannot be accessed.

        """
        n = len(train_loader)
        n_val = len(val_loader)
        # if show_train_steps is not specified then default it to print training progress 4 times per epoch
        if not show_train_steps:
            show_train_steps = n // 4 if ((n // 4) > 1) else 1

        assert (show_train_steps > 0) and (show_train_steps <= n), \
            "'show_train_steps' value-{} is out of range. " \
            "Must be >0 and <={} i.e. len(train_loader)".format(show_train_steps, n)
        assert (show_validation_epochs < num_epochs) or (num_epochs == 1), \
            "'show_validation_epochs' value should be less than 'num_epochs'"

        # reset metric dicts
        self.val_metrics = {"loss": []}
        self.train_metrics = {"loss": []}

        self.start_time = time.time()
        self.best_metric = None
        self.best_model = None
        if show_grad_flow: 
            watch_grads = WatchGrads(self.model.named_parameters())
        
        for epoch in range(num_epochs):
            # if early stopping is on, check if stop signal is switched on
            if self._stop_training:
                return self.finish_training(epoch)
            else:
                # train mode
                self.model.train()
                # 'running_loss' accumulates loss until it gets printed every 'show_train_steps'.
                # 'accumulates predictions and labels until the metrics are calculated in the epoch
                running_loss = []
                all_outputs = []
                all_labels = []

                for i, data in enumerate(train_loader):
                    inputs, labels = self.arrange_data(data, inputs_key, labels_key)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    if self.training_time_callback:
                        outputs = self.training_time_callback(
                            inputs, labels, i, epoch)
                    else:
                        outputs = self.model(inputs)
                        
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    # clip gradients to avoid exploding gradients
                    clip_grad_value_(self.model.parameters(), 0.8)
                    self.optimizer.step()

                    # update loss
                    running_loss.append(loss.item())
                    # print loss every 'show_train_steps' mini-batches
                    if i % show_train_steps == 0:
                        if i != 0:
                            print("[%d, %5d] loss: %.5f" % (epoch, i, np.mean(running_loss)))

                        # store the outputs and labels for computing metrics later     
                        all_outputs.append(outputs)
                        all_labels.append(labels)
                        # store the the grad flowing through the model layers
                        # during training for visualization later
                        if show_grad_flow:
                            watch_grads.store(self.model.named_parameters())

                # <end-of-training-cycle-loop>
                # at the end of an epoch, calculate metrics, report them and
                # store them in respective report dicts
                self._estimate_and_report_metrics(
                    all_outputs, all_labels, running_loss,
                    metrics_dict=self.train_metrics,
                    phase="train"
                )
                del all_outputs, all_labels, running_loss

                # validate every x iterations
                if epoch % show_validation_epochs == 0:
                    running_loss_val = []
                    all_outputs = []
                    all_labels = []

                    self.model.eval()

                    with torch.no_grad():
                        for i, data in enumerate(val_loader):
                            inputs, labels = self.arrange_data(data, inputs_key, labels_key)

                            # forward pass only
                            if self.training_time_callback is not None:
                                outputs = self.training_time_callback(
                                    inputs,
                                    labels,
                                    1,  # dummy value
                                    1  # dummy value
                                )
                            else:
                                outputs = self.model(inputs)
                        
                            loss = self.criterion(outputs, labels)

                            running_loss_val.append(loss.item())
                            # store the outputs and labels for computing metrics later
                            all_outputs.append(outputs)
                            all_labels.append(labels)

                    # report validation metrics
                    self._estimate_and_report_metrics(
                        all_outputs, all_labels, running_loss_val,
                        metrics_dict=self.val_metrics,
                        phase="val"
                    )
                    del all_outputs, all_labels, running_loss_val
                    # <end-of-one-epoch-loop>

                if self.scheduler:
                    self.scheduler.step()
                    
            # <end-of-all-epochs-loop>
            for callback in self.callbacks:
                callback(self, epoch=epoch)
                
        # End training
        if show_grad_flow: 
            save_fig_path = show_grad_flow if isinstance(show_grad_flow,str) else ''
            watch_grads.plot(save_fig_path=save_fig_path)
        return self.finish_training(epoch)
    

    def finish_training(self, epoch):
        """End the training cyle, return a model and finish callbacks.

        Parameters
        ----------
        epoch : int
            The current epoch.

        Returns
        -------
        tuple
            First entry is the trained model.
            Second entry is a dictionary containing:
            "train_metrics": all train_metrics
            "val_metrics": all val_metrics
            "best_model": best_model
            "best_metric": best_metric

        Raises
        ------
        AttributeError
            the 'final' function for a Callback failed.

        """
        time_elapsed = int(time.time() - self.start_time)
        print("Total time elapsed: {}h:{}m:{}s".format(
            time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

        # execute final methods of callbacks
        for callback in self.callbacks:
            # find all methods of the callback
            try:
                callback.final(trainer=self, epoch=epoch)
            except AttributeError:
                pass

        # in case of no model selection, pick the last loss
        if not self.best_metric:
            self.best_metric = self.val_metrics["loss"][-1]
            self.best_model = self.model

        return (self.model,
                {
                    "train_metrics": self.train_metrics,
                    "val_metrics": self.val_metrics,
                    "best_model": self.best_model,
                    "best_metric": self.best_metric}
                )

    
    def visualize_training(self, report, metrics=None, save_fig_path=""):
        """A function to vizualize model training.

        Parameters
        ----------
        report : dict
            must store key "train_metrics" and "val_metrics".
        metrics
            Metrics to visualize. Default: None
        save_fig_path : str
            A path to store figures in a pdf file. Default: "" (Do not plot to pdf)

        """
        # TODO: if metric is empty print so
        for metric_name in report["train_metrics"].keys():
            # if metrics is not specified, plot everything, otherwise only plot the given metrics
            if metrics is None or metric_name.split(" ")[-1] in [m.__name__ for m in metrics]:
                # todo: add x and y label description or use seaborn!
                plt.figure()
                plt.plot(report["train_metrics"][metric_name])
                plt.plot(report["val_metrics"][metric_name])
                plt.legend(["Train", "Val"])
                plt.title('Training curve: '+metric_name.replace('_',' ').title())
                if save_fig_path:
                    plt.savefig(save_fig_path+f"training_curve_{metric_name.replace(' ','')[:3]}.jpg")
                    plt.close()
                else:
                    plt.show()
                    

    def evaluate_model(
            self,
            val_loader,
            metrics=[],
            inputs_key="image",
            labels_key="label",
            write_to_dir='', 
            return_results=False
    ):
        """Predict on the validation set.

        Parameters
        ----------
        val_loader: torch.utils.DataLoader
            The data which should be used for model evaluation.
        metrics
            Metrics to assess. Default: []
        inputs_key, labels_key
            The data returned by 'val_loader' can either be a dict of format
            data_loader[X_key] = inputs and data_loader[y_key] = labels
            or a list with data_loader[0] = inputs and data_loader[1] = labels.
            The default keys are "image" and "label".
        write_to_dir
            The outputs of the evaluation are written to files path provided. Default: ""
        return_preds: If set to True, also returns model's output probabilities along with the true labels 
        """
        self.model.eval()

        running_loss = []
        all_outputs = []
        all_labels = []
        data_extras = {}

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = self.arrange_data(data, inputs_key, labels_key)
                # if the dataloader returns more info, then store and return them too
                if return_results and len(data)>2:
                    if i==0: # create entries in data_extras
                        data_extras = {k: data[k].numpy().tolist() for k in data.keys() if k not in [inputs_key, labels_key]}
                    else:
                        for k in data.keys():
                            if k not in [inputs_key, labels_key]:
                                data_extras[k].extend(data[k].numpy().tolist())
                        
                if self.training_time_callback:
                    outputs = self.training_time_callback(
                        inputs, labels, 1, 1)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())
                all_outputs.append(outputs)
                all_labels.append(labels)

            # calculate the loss criterion metric
            results = {"loss": []}

            # if new metrics are provided, update self.metrics
            if metrics:
                if self.multitask:
                    self.metrics = metrics
                else:
                    self.metrics = [metrics]

            # calculate metrics
            self._estimate_and_report_metrics(
                all_outputs, all_labels, running_loss,
                metrics_dict=results,
                phase="eval",
                save_fig_path=write_to_dir
            )

        if write_to_dir:
            results = {k: v[0] for k, v in results.items()}
            with open(write_to_dir + "eval_results.json", "w") as f:
                json.dump(results, f)
            
        self.model.train()
        
        if return_results: return all_outputs, all_labels, results, data_extras
        

    def _estimate_and_report_metrics(
            self, all_outputs, all_labels,
            running_loss, metrics_dict, phase,
            save_fig_path=""
    ):
        """ Function executed at the end of an epoch.
        Notes
        -----
            (a) calculate metrics
            (b) store results in respective report dicts
            (c) report metrics
        """

        # report execution time, only in training phase
        if phase == "train":
            time_elapsed = int(time.time() - self.start_time)
            print("Time elapsed: {}h:{}m:{}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

        if isinstance(all_outputs[0], list):
            all_outputs = [torch.cat(out).float() for out in zip(*all_outputs)]
            if self.multitask and not all([isinstance(metrics_per_task, list) for metrics_per_task in self.metrics]):
                print("WARNING: You are doing multi-task training. You should provide metrics for each "
                      "sub-task as a list of lists but a single value is provided."
                      " No metrics will be calculated for secondary tasks")
                self.metrics = [self.metrics] + [[] for _ in range(len(all_outputs))]
            if self.multitask and not isinstance(self.task_type, list):
                print("WARNING: In multi-task training, you should provide task_type "
                      " for each sub-task as a list but a single value is provided. Assuming the secondary tasks have"
                      " the same task_type '{}'!".format(self.task_type))
                self.task_type = [self.task_type for _ in range(len(all_outputs))]
        else:
            all_outputs = [torch.cat(all_outputs).float()]
        
        if isinstance(all_labels[0], list):
            all_labels = [torch.cat(lbl).float() for lbl in zip(*all_labels)]
        else:
            all_labels = [torch.cat(all_labels).float()]

        # add loss to metrics_dict
        loss = np.mean(running_loss)
        metrics_dict["loss"].append(loss)
        # print the loss for val and eval phases
        if phase in ["val", "eval"]:
            print("{} loss: {:.5f}".format(phase, loss), flush=True)

        # calculate other metrics and add to the metrics_dict for all tasks
        if self.multitask:
            for task_idx in range(len(all_outputs)):
                # perform inference on the outputs
                all_pred, all_label = predict(
                    all_outputs[task_idx],
                    all_labels[task_idx],
                    self.task_type[task_idx],
                    self._criterions[task_idx],
                    **self.kwargs
                )            
            # If it is a multi-head training then append prefix      
            metric_prefix = "task{} ".format(task_idx + 1)
            
        else:
            task_idx = 0
            # perform inference on the outputs
            all_pred, all_label = predict(
                all_outputs[task_idx],
                all_labels[task_idx],
                self.task_type[task_idx],
                self._criterions[task_idx],
                **self.kwargs
            )
            metric_prefix = ''

            # report metrics
            for metric in self.metrics[task_idx]:
                result = metric(all_label, all_pred)

                metric_name = metric_prefix + metric.__name__
                if isinstance(result, float):
                    print("{} {}: {:.2f} %".format(
                        phase, metric_name, result * 100))
                else:
                    print("{} {}: {} ".format(
                        phase, metric_name, str(result)))
                # store results in the report dict
                if metric_name in metrics_dict:
                    metrics_dict[metric_name].append(result)
                else:
                    metrics_dict[metric_name] = [result]

            # plot confusion matrix if it is a binary classification
            if phase == "eval" and self.task_type[task_idx] == "classif_binary":
                # TODO: switch to seaborn
                cm = confusion_matrix(all_label, all_pred)
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

                # Visualize the confusion matrix
                classes = ["control", "patient"]
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)

                thresh = cm.max() / 2.
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
                plt.title("Confusion Matrix")
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                
                if save_fig_path:
                    plt.savefig(save_fig_path + "eval_cmat.jpg")
                plt.show()
                

    def _extract_region(self, x, region_mask):

        region_mask = torch.from_numpy(region_mask).to(self.device)

        B, C, H, W, D = x.shape

        patch = []
        for i in range(B):
            im = x[i].unsqueeze(dim=0)
            # T = im.shape[-1]

            im = im * region_mask.float()
            # and finally extract
            patch.append(im)
        patch = torch.cat(patch)

        return patch
