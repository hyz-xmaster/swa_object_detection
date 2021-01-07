import os.path as osp

from mmcv.runner import HOOKS, Hook
from mmcv.runner.checkpoint import save_checkpoint
from torch.optim.swa_utils import AveragedModel


@HOOKS.register_module()
class SWAHook(Hook):
    r"""SWA Object Detection Hook.

        This hook works together with SWA training config files to train
        SWA object detectors <https://arxiv.org/abs/2012.12645>.

        Args:
            swa_eval (bool): Whether to evaluate the swa model.
                Defaults to True.
            eval_hook (Hook): Hook class that contains evaluation functions.
                Defaults to None.
    """

    def __init__(self, swa_eval=True, eval_hook=None):
        if swa_eval:
            assert eval_hook is not None
        self.swa_eval = swa_eval
        self.eval_hook = eval_hook

    def before_run(self, runner):
        """Construct the averaged model which will keep track of the running
        averages of the parameters of the model."""
        model = runner.model
        self.model = AveragedModel(model)

    def after_train_epoch(self, runner):
        """Update the parameters of the averaged model, save and evaluate the
        updated averaged model."""
        model = runner.model
        # update the parameters of the averaged model
        self.model.update_parameters(model)

        # save the swa model
        runner.logger.info(
            f'Saving swa model at swa-training {runner.epoch + 1} epoch')
        filename = 'swa_model_{}.pth'.format(runner.epoch + 1)
        filepath = osp.join(runner.work_dir, filename)
        optimizer = runner.optimizer
        meta = runner.meta
        meta['hook_msgs']['last_ckpt'] = filepath
        save_checkpoint(
            self.model.module, filepath, optimizer=optimizer, meta=meta)

        # evaluate the swa model
        if self.swa_eval:
            self.work_dir = runner.work_dir
            self.rank = runner.rank
            self.epoch = runner.epoch
            self.logger = runner.logger
            self.log_buffer = runner.log_buffer
            self.meta = runner.meta
            self.meta['hook_msgs']['last_ckpt'] = filename
            self.eval_hook.after_train_epoch(self)

    def after_run(self, runner):
        # since BN layers in the backbone are frozen,
        # we do not need to update the BN for the swa model
        pass

    def before_epoch(self, runner):
        pass
