import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import data_loader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pytorch_metric_learning.losses
import pytorch_metric_learning.miners
import pytorch_metric_learning.samplers

def initialize_model(use_pretrained=True, l1Units=256, l2Units=64):

    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=use_pretrained)
    for param in model.parameters():
        param.requires_grad = False  # because these layers are pretrained
    # change the final layer to be a bottle neck of two layers
    extracted_features_size = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(extracted_features_size, l1Units),
        nn.Linear(l1Units, l2Units)
    )
    return model

def train(args, model, device, train_loader, optimizer, epoch, triplet_loss_func, miner=None):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()  # Set the model to training mode
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()  # Clear the gradient
        embeddings = model(data)  # Make predictions
        if miner:
            mined_pairs = miner(embeddings, labels)
            loss = triplet_loss_func(embeddings, labels, mined_pairs)
        else:
            loss = triplet_loss_func(embeddings, labels)
        loss.backward()  # Gradient computation
        optimizer.step()  # Perform a single optimization step
        if batch_idx % args.batch_log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, dataName):
    model.eval()  # Set the model to inference mode
    test_loss = 0
    correct = 0 # number of times it gets the distances correct
    test_num = 0
    with torch.no_grad():  # For the inference step, gradient is not computed
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            embeddings = model(data)
            # TO DO: function that takes output and turns into anchor, positive, negative
            test_loss += F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2) # sum up batch loss
            # pull the predicted matches from the output
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(labels.view_as(pred)).sum().item()
            #test_num += len(data)

    #test_loss /= test_num

    # print('\n' + dataName + ' tested: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, test_num,
    #     100. * correct / test_num))


    return test_loss #, correct, test_num

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='TripNet: a network for ReID')
    parser.add_argument('--name',
                        help="what you want to name this model save file")
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--weight-decay', type=float, default=0.02, metavar='M',
                        help='Learning rate step gamma (default: 0.02)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--load-model', type=str,
                        help='model file path or model name for plotting fract comparison')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    # It might be helpful to split data_folder into separate arguments for train/val/test
    parser.add_argument('--data-folder', required=True,
                        help='folder containing data images')
    parser.add_argument('--train-json', required=True,
                        help='JSON with COCO-format annotations for training dataset')
    parser.add_argument('--val-json', required=True,
                        help='JSON with COCO-format annotations for validation dataset')
    parser.add_argument('--batch-log-interval', type=int, default=10,
                        help='Number of batches to run each epoch before logging metrics.')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(2021)  # to ensure you always get the same train/test split
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # TODO: update these (placeholder) transforms
    # Also, we may need different transforms for train/val
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize([400, 600]), # Some images are slightly different sizes
        torchvision.transforms.ToTensor(),
    ])

    # Initialize dataset loaders
    train_dataset = data_loader.CocoDataset(
        args.data_folder, args.train_json, transforms
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=pytorch_metric_learning.samplers.MPerClassSampler(
            [train_dataset.coco.anns[annotation_id]['name'] for annotation_id in train_dataset.annotation_ids],
            m=2,
            batch_size=args.batch_size,
            length_before_new_iter=len(train_dataset)
        ),
        batch_size=args.batch_size,
        num_workers=4
        )
    val_loader = data_loader.get_loader(
        args.data_folder,
        args.val_json,
        transforms,
        batch_size=args.batch_size,
        shuffle=True
    )

    # object recognition, pretrained on imagenet
    # https://pytorch.org/hub/pytorch_vision_densenet/
    model = initialize_model(use_pretrained=True)
    print(model)
    model = model.to(device)
    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    trainLoss = []
    valLoss = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,
                triplet_loss_func=pytorch_metric_learning.losses.TripletMarginLoss(
                    margin=0.2, smooth_loss=True, triplets_per_anchor=10,
                ),
                miner=pytorch_metric_learning.miners.TripletMarginMiner()
            )
        trloss = test(model, device, train_loader, "train data")
        vloss = test(model, device, val_loader, "val data")
        trainLoss.append(trloss)
        valLoss.append(vloss)
        scheduler.step()  # learning rate scheduler

    if args.save_model:
            torch.save(model.state_dict(), args.name + "_model.pt")

    # plot training and validation loss by epoch
    f = plt.figure(figsize=(6, 5))
    ax = plt.subplot()
    plt.plot(range(1, args.epochs + 1), trainLoss, label="training loss")
    plt.plot(range(1, args.epochs + 1), valLoss, label="validation loss")
    ax.set_title('loss over epochs')
    # plt.xlim([0, 1.1])
    # plt.ylim([0, 1.1])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


# feed into the network triplets of zebras, with segmented out backgrounds

if __name__ == '__main__':
    main()
