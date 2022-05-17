from scripts.settr_pl import SetTrModel
from scripts.config import build_parser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

parser = build_parser()
config = parser.parse_args()
seed_everything(config.np_seed)

wandb.init(project=config.exp_project_name, entity="aleksey-postnikov", name=config.exp_name)
wandb.config = {
    "learning_rate": config.exp_lr,
    "epochs": config.exp_num_epochs,
    "batch_size": config.exp_batch_size
}


# model = AttPredictorPecNetWithTypeD3(config=config, wandb_logger=wandb)
model = SetTrModel(config, wandb_logger=wandb)
# model = SimpleModel.load_from_checkpoint("/home/jovyan/waymo-open-dataset/tutorial/lightning_logs/version_61/checkpoints/epoch=3-step=3415.ckpt", config=config, wandb_logger=wandb)

lr_monitor = LearningRateMonitor(logging_interval='step')

# trainer = Trainer(overfit_batches=2, accelerator="gpu", max_epochs=100, devices=1)#, num_sanity_val_steps=0 
# trainer.fit(model)
trainer = Trainer(accelerator="gpu", callbacks=[lr_monitor], max_epochs=100, devices=-1, strategy="dp", check_val_every_n_epoch=1)#, num_sanity_val_steps=0

# call tune to find the lr
# trainer.tune(model)
trainer.fit(model)
trainer.validate(model)
# trainer = Trainer(devices=2, accelerator="gpu")
# trainer = Trainer(accelerator="gpu", callbacks=[lr_monitor])

pass
