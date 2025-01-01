
import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time
from model import Model
from metrics import DiceLoss, DiceBCELoss

def load_data(path):
    def get_data(path, name):
        images = sorted(glob(os.path.join(path, name, "images", "*.jpg")))
        labels = sorted(glob(os.path.join(path, name, "masks", "grayscale", "liver", "*.jpg")))
        return images, labels

    """ Names """
    dirs = sorted(os.listdir(path))
    test_names = [f"liver_{i}" for i in range(0, 21, 1)]
    valid_names = [f"liver_{i}" for i in range(21, 41, 1)]

    print(test_names)
    print(valid_names)

    train_names = [item for item in dirs if item not in test_names]
    train_names = [item for item in train_names if item not in valid_names]

    """ Training data """
    train_x, train_y = [], []
    for name in train_names:
        x, y = get_data(path, name)
        train_x += x
        train_y += y

    """ Validation data """
    valid_x, valid_y = [], []
    for name in valid_names:
        x, y = get_data(path, name)
        valid_x += x
        valid_y += y

    """ Testing data """
    test_x, test_y = [], []
    for name in test_names:
        x, y = get_data(path, name)
        test_x += x
        test_y += y

    return [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, self.size)
        mask = mask/255.0
        mask = (mask >= 0.5).astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred = model(x)
        loss = 0.0
        for p in pred:
            loss += loss_fn(p, y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            pred = model(x)
            loss = 0.0
            for p in pred:
                loss += loss_fn(p, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
        return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 512
    size = (image_size, image_size)
    num_classes=1
    batch_size = 8
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "files/checkpoint.pth"
    path = "../ML_DATASET/MSD-2D/Task03_Liver"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    # train_x, train_y = train_x[:1000], train_y[:1000]
    # valid_x, valid_y = valid_x[:1000], valid_y[:1000]

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    # create_dir("data")
    # for i, (x, y) in enumerate(train_dataset):
    #     x = np.transpose(x, (1, 2, 0)) * 255
    #     y = np.expand_dims(y, axis=-1) * 255
    #     y = np.concatenate([y, y, y], axis=-1)
    #     cv2.imwrite(f"data/{i}.png", np.concatenate([x, y], axis=1))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = Model(image_size=image_size, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "DL + BCE"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_loss_metrics = float("inf")
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_loss < best_loss_metrics:
            data_str = f"Valid loss improved from {best_loss_metrics:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_loss_metrics = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_loss > best_loss_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
