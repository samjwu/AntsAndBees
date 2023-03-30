from __future__ import division, print_function

import copy
import numpy
import matplotlib.pyplot
import os
import time
import torch
import torch.backends.cudnn
import torch.nn
import torch.optim
import torchvision

def show_image(input_data, title=None):
    input_data = input_data.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.485, 0.456, 0.406])
    standard_deviation = numpy.array([0.229, 0.224, 0.225])

    input_data = input_data * standard_deviation + mean
    input_data = numpy.clip(input_data, 0, 1)

    matplotlib.pyplot.imshow(input_data)
    if title is not None:
        matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct_answers = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs) # forward
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct_answers += torch.sum(predictions == labels.data)

            if phase == 'training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_correct_answers.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')

            if phase == 'validation' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_accuracy:4f}')

    model.load_state_dict(best_weights)
    return model

if __name__ ==  '__main__':
    data_transforms = {
        'training': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    torch.backends.cudnn.benchmark = True

    data_dir = 'data'
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['training', 'validation']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        for x in ['training', 'validation']}
    dataset_sizes = {x: len(image_datasets[x]) 
        for x in ['training', 'validation']}
    class_names = image_datasets['training'].classes

    if torch.cuda.is_available():
        print("Using CUDA-enabled GPU")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # get a batch of training data
    inputs, classes = next(iter(dataloaders['training']))

    out = torchvision.utils.make_grid(inputs)

    show_image(out, title=[class_names[x] for x in classes])
