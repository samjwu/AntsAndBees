"""
Transfer learning for computer vision example. 
Convolutional neural network for classifying ant and bee images.
"""
from __future__ import division, print_function

import os
import time
import copy
import numpy
import matplotlib.pyplot
import torch
import torch.backends.cudnn
import torch.nn
import torch.optim
import torchvision

def show_image(input_data, title=None):
    """
    Visualize data augmentations
    """
    input_data = input_data.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.485, 0.456, 0.406])
    standard_deviation = numpy.array([0.229, 0.224, 0.225])

    input_data = input_data * standard_deviation + mean
    input_data = numpy.clip(input_data, 0, 1)

    matplotlib.pyplot.imshow(input_data)
    if title is not None:
        matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    """
    Train model on training dataset to reduce loss
    Save a deep copy of the best model weights based on accuracy during validation
    """
    start = time.time()

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

    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_accuracy:4f}')

    model.load_state_dict(best_weights)
    return model

def visualize_model_predictions(model, num_images=6):
    """
    Display predictions for num_images
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    matplotlib.pyplot.figure()

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = matplotlib.pyplot.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[predictions[j]]}')
                show_image(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

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
    DATA_DIR = 'data'

    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(DATA_DIR, x),
        data_transforms[x])
        for x in ['training', 'validation']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
        batch_size=4, shuffle=True, num_workers=4)
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

    output = torchvision.utils.make_grid(inputs)

    show_image(output, title=[class_names[x] for x in classes])

    cnn_model = torchvision.models.resnet18(weights='DEFAULT')
    for parameter in cnn_model.parameters():
        parameter.requires_grad = False

    num_features = cnn_model.fc.in_features

    cnn_model.fc = torch.nn.Linear(num_features, len(class_names))

    cnn_model = cnn_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer_features = torch.optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

    # decay learning rate by a factor of gamma every step_size epochs
    exponential_learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_features,
        step_size=7,
        gamma=0.1
    )

    cnn_model = train_model(
        cnn_model,
        criterion,
        optimizer_features,
        exponential_learning_rate_scheduler,
        num_epochs=10
    )

    visualize_model_predictions(cnn_model)
