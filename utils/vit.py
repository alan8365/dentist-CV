import os
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import ImageDataset, create_loader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import pad


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
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
            pred = torch.sigmoid(pred)

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
    precision = float('nan') if tp + fp == 0 else tp / (tp + fp)
    sensitivity = float('nan') if tp + fn == 0 else tp / (tp + fn)

    # print("Test Error:")
    # print(f"Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    # print(f"Precision: {(100 * precision):>0.1f}%, Sensitivity: {(100 * sensitivity):>0.1f}% \n")

    writer.add_scalar('Loss/test', test_loss, epoch)

    writer.add_scalars('Accuracy/test', {'all': accuracy}, epoch)
    writer.add_scalars('Precision/test', {'all': precision}, epoch)
    writer.add_scalars('Sensitivity/test', {'all': sensitivity}, epoch)

    for idx, col in enumerate(torch.transpose(confusion_table, 0, 1)):
        tp, fp, tn, fn = col

        accuracy = (tp + tn) / col.sum()
        precision = float('nan') if tp + fp == 0 else tp / (tp + fp)
        sensitivity = float('nan') if tp + fn == 0 else tp / (tp + fn)

        classes_name = classes[idx]
        # print(f'Class {classes_name}:')
        # print(f"Accuracy: {(100 * accuracy):>0.1f}%")
        # print(f"Precision: {(100 * precision):>0.1f}%, Sensitivity: {(100 * sensitivity):>0.1f}%")

        writer.add_scalars('Accuracy/test', {classes_name: accuracy}, epoch)
        writer.add_scalars('Precision/test', {classes_name: precision}, epoch)
        writer.add_scalars('Sensitivity/test', {classes_name: sensitivity}, epoch)


def evaluation(dataloader, model, loss_fn, classes_size, threshold=0.5, device=torch.device('cpu'),
               writer: SummaryWriter = None,
               classes=None):
    # TODO PR curve
    # TODO confusion matrix
    pass

    idx = 1
    classes_name = classes[idx]

    # writer.add_pr_curve(classes_name, )


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


def predict():
    model_dir = Path(os.getenv('ViT_MODEL_DIR'))
    temp_dir = Path(os.getenv('TEMP_DIR')) / 'crop_tooth_image'

    # Vit model loading
    model_path = model_dir / 'yolov8-base.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vit_model = timm.create_model('swin_base_patch4_window7_224_in22k', num_classes=6)
    vit_model.load_state_dict(torch.load(model_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    # Preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
        # (lambda image: padding_to_size(image, 224)),
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    target_transform = transforms.Compose([
        (lambda y: torch.Tensor(y)),
    ])
    dataset = ImageDataset(temp_dir, transform=transform)

    if torch.cuda.is_available():
        dataloader = create_loader(dataset, (3, 224, 224), 4)
    else:
        dataloader = create_loader(dataset, (3, 224, 224), 4, use_prefetcher=False)

    size = len(dataloader.dataset)

    threshold = torch.Tensor([0.5, 0.85, 0.5, 0.5, 0.5, 0.5]).to(device)
    pred_encodes = []
    # target_labels = ['caries', 'endo', 'post', 'crown']
    target_labels = ['R.R', 'caries', 'crown', 'endo', 'filling', 'post']
    # target_labels = ['caries', 'crown', 'endo', 'filling', 'post']
    with torch.no_grad():
        for batch, (X, _) in enumerate(dataloader):
            X = X.to(device)

            # Compute prediction error
            pred = vit_model(X)
            pred_encode = pred > threshold
            pred_encodes.append(pred_encode.cpu().numpy())

    pred_encodes = np.vstack(pred_encodes)
    pred_encodes = pred_encodes[:, 1:]
    detected_list = [()] * len(pred_encodes)
    for im_path, pred_encode in enumerate(pred_encodes):
        detected_list[im_path] = tuple((target_labels[j] for j, checker in enumerate(pred_encode) if checker))

    return detected_list


if __name__ == "__main__":
    a = (lambda: np.random.randint(1, 100))

    imgs = [torch.tensor(np.ones((1, a(), a()))) for _ in range(10)]
    org_shapes = [img.shape for img in imgs]
    pad_shapes = [padding_to_size(img, 200).shape for img in imgs]
