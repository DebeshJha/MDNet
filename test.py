
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import f1_score, jaccard_score
from model import Model
from utils import create_dir, seeding, calculate_metrics
from train import load_data

def overlay_mask(ct_image, mask_rgb, color=(0, 0, 255), alpha=0.5):
    mask = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    mask_rgb[mask > 128] = color
    overlay = cv2.addWeighted(mask_rgb, alpha, ct_image, 1 - alpha, 0)
    return overlay

def process_mask(y_pred, size):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = cv2.resize(y_pred, size)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def print_score(metrics_score):
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)
    hd = metrics_score[6]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f} - HD: {hd:1.4f}")

def evaluate(model, save_path, test_x, test_y, size):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    metrics_score_2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    metrics_score_3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")
        dir_name = name[-5]
        image_name = name[-1]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()

            [p1, p2, p3] = model(image)
            p1 = torch.sigmoid(p1)
            p2 = torch.sigmoid(p2)
            p3 = torch.sigmoid(p3)

            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            score_2 = calculate_metrics(mask, p2)
            score_3 = calculate_metrics(mask, p3)

            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            metrics_score_2 = list(map(add, metrics_score_2, score_2))
            metrics_score_3 = list(map(add, metrics_score_3, score_3))

            """ Predicted Mask """
            p1 = process_mask(p1, size)
            p2 = process_mask(p2, size)
            p3 = process_mask(p3, size)

        """ Save the image - mask - pred """
        create_dir(f"{save_path}/{dir_name}/joint")
        create_dir(f"{save_path}/{dir_name}/mask")
        create_dir(f"{save_path}/{dir_name}/overlay")
        create_dir(f"{save_path}/{dir_name}/mask_overlay")

        line = np.ones((size[1], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, p1, line, p2, line, p3], axis=1)
        cv2.imwrite(f"{save_path}/{dir_name}/joint/{image_name}", cat_images)
        cv2.imwrite(f"{save_path}/{dir_name}/mask/{image_name}", p3)
        cv2.imwrite(f"{save_path}/{dir_name}/mask_overlay/{image_name}", overlay_mask(save_img, p3))

        overlay_images = np.concatenate([
            save_img, line,
            overlay_mask(save_img, save_mask), line,
            overlay_mask(save_img, p1), line,
            overlay_mask(save_img, p2), line,
            overlay_mask(save_img, p3)

        ], axis=1)
        cv2.imwrite(f"{save_path}/{dir_name}/overlay/{image_name}", overlay_images)

    print_score(metrics_score_1)
    print_score(metrics_score_2)
    print_score(metrics_score_3)

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    image_size = 512
    size = (image_size, image_size)
    num_classes=1

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(image_size=image_size, num_classes=num_classes)
    model = model.to(device)
    checkpoint_path = "files/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "../ML_DATASET/MSD-2D/Task03_Liver"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)

    save_path = f"results/Task03_Liver"
    create_dir(save_path)
    evaluate(model, save_path, test_x, test_y, size)
