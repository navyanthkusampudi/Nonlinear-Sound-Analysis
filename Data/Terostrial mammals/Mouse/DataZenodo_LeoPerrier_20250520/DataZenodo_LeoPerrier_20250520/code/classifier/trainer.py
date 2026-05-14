import time

import torch


def train_model(model, device, dataloaders, writer_dict, exp_ref, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    results = f""
    best_model_params_path = f"models/best_model_{exp_ref}.pth"
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        results += f"Epoch {epoch+1}/{num_epochs}\n{'-' * 10}\n"

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            results += f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n'

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

            writer = writer_dict[f'{phase}_writer']
            writer.add_scalar("loss", epoch_loss, epoch)
            writer.add_scalar("accuracy", epoch_acc, epoch)

        results += f"\n\n"
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    results += f"\n\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    results += f"\nBest val Acc: {best_acc:4f} \n\n"

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))

    return model, results
