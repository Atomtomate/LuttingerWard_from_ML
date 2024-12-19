from pytorch_lightning.callbacks import Callback

class TestLogger(Callback):
    def __init__(self):
        super().__init__()
        self.test_loss = []

    def on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Extract metrics from the trainer's logger after the validation epoch ends
        #test_loss = trainer.callback_metrics.get('test/loss', None)
        print(f"aaaa: {len(outputs['test/acc'])}")
        #if test_loss is not None:
        #    self.val_losses.append(val_loss.item())
        pass