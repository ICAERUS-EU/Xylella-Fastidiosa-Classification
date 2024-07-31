from torch.utils.data import DataLoader
from torchvision import transforms
from classifier import DualYOLO, CustomDataset
import torch
import pandas as pd


if __name__ == "__main__":

    csvs_path = './dataset/labels'
    ndvi_path = './dataset/images/ndvi'

    rgb_path = './dataset/images/rgb'
    save_dir_validation_set='./runs/classified/val'

    checkpoint_path = './training_results/models/version_0/epoch=28-val_loss=0.63.ckpt'

    model = DualYOLO.from_pretrained(checkpoint_path, depth_scale=0.67, width_scale=0.75)
    model = model.cuda()

    model.eval()
    validation_set_path='./runs/classified/val/validation_set.csv'

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    val_dataset = CustomDataset(health_dir_path=validation_set_path, ndvi_path=ndvi_path, rgb_path=rgb_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=11)

    predictions = []
    image_paths_img1 = []
    image_paths_img2 = []
    with torch.no_grad():
        for idx, (img1, img2, _) in enumerate(val_loader):
            img1, img2 = img1.cuda(), img2.cuda()
            outputs = model(img1, img2)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

            start_idx = idx * val_loader.batch_size
            end_idx = start_idx + img1.size(0)
            image_paths_img1.extend(val_dataset.data_img1_paths[start_idx:end_idx])
            image_paths_img2.extend(val_dataset.data_img2_paths[start_idx:end_idx])

    results_df = pd.DataFrame({
        'predicted_label': predictions,
        'image_path_img1': image_paths_img1,
        'image_path_img2': image_paths_img2
    })

    results_df.to_csv('./runs/classified/val/validation_predictions.csv', index=False)
