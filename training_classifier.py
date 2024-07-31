import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torchvision import transforms
from classifier import CustomDataset,ValDataset, DualYOLO

if __name__ == "__main__":
    # Define image directories
    csvs_path = './dataset/labels'
    ndvi_path = './dataset/images/ndvi'

    rgb_path = './dataset/images/rgb'
    save_dir_validation_set='./runs/classified/val'

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(health_dir_path=csvs_path, ndvi_path=ndvi_path, rgb_path=rgb_path, val_samples_per_file=400, transform=transform,save_dir=save_dir_validation_set)
    val_dataset = ValDataset(dataset)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=11)

    # Shuffle the validation set is important for good calculation of validation metrics
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=11)

    train_labels = dataset.get_train_labels()

    # Compute weights classes, training dataset could be unbalanced
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda')

    save_dir = './training_results/models'

    # Initialize DualYOLO model
    dual_yolo_model = DualYOLO(class_weights,depth_scale=0.67, width_scale=0.75, lr=1e-6,num_classes=3,save_dir=save_dir)
    # Define the directory to save the metrics and model weights

    os.makedirs(save_dir, exist_ok=True)

    # Ensure the logger directory exists
    logger_dir = './training_results/logs'
    os.makedirs(logger_dir, exist_ok=True)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name='metrics')

    # CSV logger
    csv_logger = CSVLogger(save_dir=save_dir, name='metrics')

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        save_top_k=20,
        mode='min'
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        precision="16-mixed",
        callbacks=checkpoint_callback,
        logger=[tb_logger, csv_logger])
    # Set CUDA_LAUNCH_BLOCKING=1 for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Train the model
    trainer.fit(dual_yolo_model, train_loader, val_loader)