import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.transforms.functional import pad

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from vit_pytorch.vit_for_small_dataset import ViT
from tqdm import tqdm


def train(dataloader, model, loss_fn, optimizer, writer: SummaryWriter = None, epoch=0, device=torch.device('cpu')):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            writer.add_scalar('Loss/train', loss, epoch)


def test(dataloader, model, loss_fn, classes_size, threshold=0.5, device=torch.device('cpu'), writer=None, epoch=0,
         classes=None):
    if classes is None:
        classes = []

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    confusion_table = torch.zeros((4, classes_size)).type(torch.int).to(
        device)  # row is order by tp, fp, tn, fn

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            pred_encode = pred > threshold
            y_encode = y.type(torch.bool)

            xor_table = pred_encode ^ y_encode

            tp_table = pred_encode & y_encode
            tn_table = torch.logical_not(pred_encode | y_encode)
            fn_table = xor_table & y_encode
            fp_table = xor_table & pred_encode

            temp_table = torch.stack((
                tp_table.sum(axis=0),
                fp_table.sum(axis=0),
                tn_table.sum(axis=0),
                fn_table.sum(axis=0),
            ))

            confusion_table += temp_table

    test_loss /= num_batches

    tp, fp, tn, fn = confusion_table.sum(axis=1)
    accuracy = (tp + tn) / confusion_table.sum()
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    sensitivity = 0 if tp + fn == 0 else tp / (tp + fn)

    print("Test Error:")
    print(f"Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"Precision: {(100 * precision):>0.1f}%, Sensitivity: {(100 * sensitivity):>0.1f}% \n")

    writer.add_scalar('Loss/test', test_loss, epoch)

    writer.add_scalars('Accuracy/test', {'all': accuracy}, epoch)
    writer.add_scalars('Precision/test', {'all': precision}, epoch)
    writer.add_scalars('Sensitivity/test', {'all': sensitivity}, epoch)

    for idx, col in enumerate(torch.transpose(confusion_table, 0, 1)):
        tp, fp, tn, fn = col

        accuracy = (tp + tn) / col.sum()
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        sensitivity = 0 if tp + fn == 0 else tp / (tp + fn)

        classes_name = classes[idx]
        print(f'Class {classes_name}:')
        print(f"Accuracy: {(100 * accuracy):>0.1f}%")
        print(f"Precision: {(100 * precision):>0.1f}%, Sensitivity: {(100 * sensitivity):>0.1f}%")

        writer.add_scalars('Accuracy/test', {classes_name: accuracy}, epoch)
        writer.add_scalars('Precision/test', {classes_name: precision}, epoch)
        writer.add_scalars('Sensitivity/test', {classes_name: sensitivity}, epoch)


def padding_to_size(image, size):
    target_height, target_width = pair(size)
    _, image_height, image_width = image.size()

    assert image_height < target_height or image_width % target_width, 'Target size is smaller than origin image size.'

    diff_height, diff_width = target_height - image_height, target_width - image_width
    diff_height, diff_width = diff_height // 2, diff_width // 2

    if target_height % 2 != image_height % 2:
        diff_height += 1
    if target_width % 2 != image_width % 2:
        diff_width += 1

    result = pad(image, [diff_width, diff_height])

    if target_height % 2 != image_height % 2:
        result = result[:, :-1, :]
    if target_width % 2 != image_width % 2:
        result = result[:, :, :-1]

    return result


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


if __name__ == "__main__":
    a = (lambda: np.random.randint(1, 100))

    imgs = [torch.tensor(np.ones((1, a(), a()))) for _ in range(10)]
    org_shapes = [img.shape for img in imgs]
    pad_shapes = [padding_to_size(img, 200).shape for img in imgs]
