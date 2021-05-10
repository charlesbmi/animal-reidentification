import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import plt

def initialize_model(use_pretrained=True, l1Units = 500, l2Units=128)

    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet201', pretrained=use_pretrained)
    for param in model.parameters():
        param.requires_grad = False  # because these layers are pretrained
    # change the final layer to be a bottle neck of two layers
    model.classifier = nn.Sequential(nn.Linear(1920, l1Units), nn.Linear(l1Units,
                                                                     l2Units))  # assuming that the fc7 layer has 512 neurons, otherwise change it
    return model

def train(args, model, device, train_loader, optimizer, epoch, triplet_loss):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear the gradient
        output = model(data)  # Make predictions
        loss = F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2)
        loss.backward()  # Gradient computation
        optimizer.step()  # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, dataName):
    model.eval()  # Set the model to inference mode
    test_loss = 0
    correct = 0 # number of times it gets the distances correct
    test_num = 0
    with torch.no_grad():  # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # TO DO: function that takes output and turns into anchor, positive, negative
            test_loss += F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2) # sum up batch loss
            # pull the predicted matches from the output
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
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
    parser.add_argument('name',
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
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(2021)  # to ensure you always get the same train/test split
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # dataset = ???

    # object recognition, pretrained on imagenet
    # https://pytorch.org/hub/pytorch_vision_densenet/
    model = initialize_model(use_pretrained=True, l1Units = 500, l2Units=128)
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
        train(args, model, device, train_loader, optimizer, epoch)
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