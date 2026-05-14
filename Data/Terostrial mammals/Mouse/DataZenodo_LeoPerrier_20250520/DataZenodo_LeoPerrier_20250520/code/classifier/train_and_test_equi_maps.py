import argparse
import torch
import datetime
import numpy as np
import torchvision
import json
import os
import cv2

from dataset_equi import create_datasets, SpectrogramDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from trainer import train_model
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.models.feature_extraction import create_feature_extractor
# from torchcam.methods import CAM
# from torchcam.utils import overlay_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--path',
                        help='path to the folder containing the spectro.',
                        required=True,
                        type=str)

    parser.add_argument('--epoch',
                        help='number of epoch to train the model',
                        type=int,
                        default=30)

    parser.add_argument('--seq',
                        help='len of the sequence, aka the number of spectro to concatenate as input',
                        type=int,
                        default=1)

    parser.add_argument('--split',
                        help='% of train, val, and test',
                        type=int,
                        nargs='+',
                        required=True)

    parser.add_argument('--batch',
                        help='batch size',
                        type=int,
                        default=64)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    for d in ["log", "datasets", "models", "results"]:
        if d not in os.listdir():
            os.makedirs(d)
    exp_ref = f"{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}_{args.epoch}_{args.seq}"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Code running on", device)

    datasets = create_datasets(args.path, args.split, args.seq)
    with open(f'datasets/{exp_ref}.json', 'w') as fp:
        json.dump(datasets, fp, sort_keys=True, indent=4)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*args.seq, std=[0.5]*args.seq)
    ])

    train_dataset = SpectrogramDataset(datasets["train"], args.seq, transform=transform)
    val_dataset = SpectrogramDataset(datasets["val"], args.seq, transform=transform)
    test_dataset = SpectrogramDataset(datasets["test"], args.seq, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    data_loader = {"train": train_loader, "val": val_loader}

    writer_dict = {
        "train_writer": SummaryWriter(log_dir=f"log/{exp_ref}/train"),
        "val_writer": SummaryWriter(log_dir=f"log/{exp_ref}/val")
    }
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = torchvision.models.resnet18(weights=None)
    model.conv1 = torch.nn.Conv2d(args.seq, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, datasets["nb_subjects"], bias=True)
    fe = create_feature_extractor(model, {'layer4': 'feat4'})

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    model.to(device)


    model, results = train_model(model, device, data_loader, writer_dict, exp_ref, criterion, optimizer, exp_lr_scheduler,
                                 num_epochs=args.epoch)

    writer_dict['train_writer'].close()
    writer_dict['val_writer'].close()

    # Test
    print("\nTesting...")
    results += f"\nTesting...\n"
    model.eval()
    # cam_extractor = CAM(model,'layer4')
    # cam_transform = transforms.ToPILImage()

    y_true = []
    y_pred = []
    cams = []
    act_maps = []
    
    os.makedirs(f'activation_maps/upscaled/{exp_ref}', exist_ok=True)
    os.makedirs(f'activation_maps/upscaled/{exp_ref}/max_activation', exist_ok=True)
    os.makedirs(f'activation_maps/upscaled/{exp_ref}/max_pooling', exist_ok=True)
    
    j = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            act_map = fe(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # cam = cam_extractor(class_idx=labels.item())
            # cams.extend(cam)
            temp = np.array(act_map['feat4'].cpu())
            up_temp = np.squeeze(np.transpose(temp,(0,2,3,1)))
            up_act_map = np.array(cv2.resize(up_temp, dsize=(inputs.size(2), inputs.size(3)),interpolation=cv2.INTER_CUBIC))
            
            # on garte la up_map la plus activée
            max_act_value = np.max(up_act_map, axis=(0, 1, 2))
            max_index = np.where(up_act_map == max_act_value)
            max_up_am = up_act_map[:, :, max_index[2][0]]
            
            # on fait un MaxPooling
            maxpooled_up_am = np.max(up_act_map, axis = 2)
            
            # temp = [temp, predicted.item()==labels.item()]
            temp = [labels.item(), predicted.item(), temp, predicted.item()==labels.item()]
            up_act_map = [labels.item(), predicted.item(), max_up_am, predicted.item()==labels.item()]
            maxpooled_act_map = [labels.item(), predicted.item(), maxpooled_up_am, predicted.item()==labels.item()]
            act_maps.append(temp)
            
            # conversion pour enregistrer en .npy
            up_act_map = np.array(up_act_map, dtype=object)
            np.save(file = f"activation_maps/upscaled/{exp_ref}/max_activation/{exp_ref}_up_am_{j}.npy", arr = up_act_map)
            maxpooled_act_map = np.array(maxpooled_act_map, dtype = object)
            np.save(file = f"activation_maps/upscaled/{exp_ref}/max_pooling/{exp_ref}_maxpooled_up_am_{j}.npy", arr = maxpooled_act_map)

            j = j + 1
            
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    act_maps = np.array(act_maps,dtype=object)
   

    np.save(file = f"activation_maps/output/{exp_ref}_am.npy", arr=act_maps)
    

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute the accuracy
    accuracy = accuracy_score(y_true, y_pred)
    results += f"Accuracy: {accuracy*100:05.2f}\n"
    print(f"Accuracy: {accuracy*100:05.2f}")
    # Compute the F1 score
    f1 = f1_score(y_true, y_pred, average='macro')
    results += f"F1-Score: {f1 * 100:05.2f}\n\n"
    print(f"F1-Score: {f1 * 100:05.2f}")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results += f"Confusion matrix: \n{cm}"
    print(f"Confusion matrix: \n{cm}")
    np.save(file = f"confusion_matrixes/{exp_ref}_cm.npy", arr=cm)

    with open(f"results/{exp_ref}_scores.txt", "w") as text_file:
        print(results, file=text_file)


if __name__ == '__main__':
    main()
