import torch
import torch.nn.functional as F
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import data_loader_triplet_v2 as data_loader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE
from PIL import Image
import pycocotools.mask as mask_util
import logging

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S')


def initialize_model(use_pretrained=True, l1Units = 500, l2Units=128):

    model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=use_pretrained)
    for param in model.parameters():
        param.requires_grad = False  # because these layers are pretrained
    # change the final layer to be a bottle neck of two layers
    extracted_features_size = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(extracted_features_size, l1Units), nn.Linear(l1Units,
                                                                     l2Units))  # assuming that the fc7 layer has 512 neurons, otherwise change it
    return model

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()  # Set the model to training mode
    for batch_idx, (img1, img2, img3), (ann1, ann2, ann3) in enumerate(train_loader):
        img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
        optimizer.zero_grad()  # Clear the gradient
        anchor_emb = model(img1)
        positive_emb = model(img2)
        negative_emb = model(img3) 
        loss = F.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=1.0, p=2)  # sum up batch loss
        loss.backward()  # Gradient computation
        optimizer.step()  # Perform a single optimization step
        if batch_idx % args.batch_log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img1), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, dataName):
    model.eval()  # Set the model to inference mode
    test_loss = 0
    correct = 0 # number of times it gets the distances correct
    test_num = 0
    with torch.no_grad():  # For the inference step, gradient is not computed
        for (img1, img2, img3), (ann1, ann2, ann3) in test_loader:
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            anchor_emb = model(img1)
            positive_emb = model(img2)
            negative_emb = model(img3) 
            # function that takes output and turns into anchor, positive, negative
            test_loss += F.triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=1.0, p=2) # sum up batch loss

    test_loss /= test_num

    logging.info('\n' + dataName + ' tested: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss #, correct, test_num

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='TripNet: a network for ReID')
    parser.add_argument('--name', default='model',
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
    parser.add_argument('--num-train-triplets', type=int, default=10*1000,
                        help='Number of triplets to generate for each training epoch.')
    parser.add_argument('--use-seg',action='store_true', default=False,
                        help='For using semantic segmentations')
    parser.add_argument('--evaluate', action='store_true', default = False,
                        help='For evaluating model performance after training')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_seg = args.use_seg
    logging.info('use seg? {}'.format(use_seg))
    np.random.seed(2021)  # to ensure you always get the same train/test split
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.evaluate:
        print('EVALUATING MODEL')
        # generate some plots, don't actually train the model
        modelName = args.name + '_model.pt'
        model = initialize_model(use_pretrained=True, l1Units = 500, l2Units=128)
        model = model.to(device)
        model.load_state_dict(torch.load(modelName))
        model.eval()

        # load the underlying annotations file for the
        BOX_ANNOTATION_FILE = '../../Data/gzgc.coco/masks/instances_train2020_maskrcnn.json'
        with open(BOX_ANNOTATION_FILE) as f:
            annData = json.load(f)
        f.close()
        annData = annData['annotations'] #just the annotations

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize([500, 750]),  # Some images are slightly different sizes
            torchvision.transforms.ToTensor(),
        ])

        val_loader = data_loader.get_loader(
            args.data_folder,
            args.val_json,
            transforms,
            batch_size=args.batch_size,
            shuffle=True,
            num_triplets=int(0.15 * args.num_train_triplets),
            apply_mask=use_seg,
        )

        # plot 6 errors on the validation set and count total number of errors
        wrong_trip = []
        wrong_posDis = []
        wrong_negDis = []
        totError = 0.0
        totCount = 0
        allFeat = np.zeros((len(val_loader.dataset), 128))
        allLabel = np.zeros((len(val_loader.dataset)))
        with torch.no_grad():  # For the inference step, gradient is not computed
            for (img1, img2, img3), (ann1, ann2, ann3) in val_loader:
                img1Dev, img2Dev, img3Dev = img1.to(device), img2.to(device), img3.to(device)
                anchor_emb = model(img1Dev) # just use these
                positive_emb = model(img2Dev)
                negative_emb = model(img3Dev)
                lastcount = totCount
                # find the errors
                for i, anc in enumerate(anchor_emb):
                    if np.linalg.norm(anc.cpu() - positive_emb.cpu()[i]) >= np.linalg.norm(anc.cpu() - negative_emb.cpu()[i]):
                        wrong_trip.append([img1[i], img2[i], img3[i]]) # anchor, positive, negative
                        wrong_posDis.append(np.linalg.norm(anc.cpu() - positive_emb.cpu()[i]))
                        wrong_negDis.append(np.linalg.norm(anc.cpu() - negative_emb.cpu()[i]))
                        totError +=1.0
                    totCount += 1

            # loop through all annotation anchors in the validation set and get the embeddings for each one
            # also make a list of names corresponding to each anchor
            zebIDs = []
            allIms = []
            allEmbeds = []
            for annID in val_loader.dataset.annotations.keys(): # for every zebraID in the dataset
                # get the image associated
                print(val_loader.dataset.annotations[annID]['image_id'])
                this_im = val_loader.dataset.images[val_loader.dataset.annotations[annID]['image_id']]
                print(this_im) # this is a dictionary entry of details about the image, not the image itself

                # TO DO want the images with the transformations
                # image_path = os.path.join('../../Data/gzgc.coco/images/train2020', this_im['file_name'])
                # # Load image
                # image = Image.open(image_path).convert('RGB')
                #
                # # Apply segmentation mask
                # if use_seg:
                #     mask = mask_util.decode(val_loader.dataset.annotations[annID]['maskrcnn_mask_rle'])
                #     segImage = image.copy()
                #     segImage = np.array(segImage)
                #     binaryMask = (mask > 0.5).astype(np.float32)
                #     segImage[np.where(binaryMask == 0.0)] = 0
                #     image = Image.fromarray(np.uint8(segImage)).convert('RGB')

                allIms.append(image.copy())
                #get the zebra name associated
                zebIDs.append(val_loader.dataset.annotations[annID]['name'])
                # run the embeddings for each image
                allEmbeds.append(model(image.to(device)))

        print(len(wrong_trip))
        print(len(wrong_trip[0]))
        print('************')
        print('total error: ' + str(totError) + '/' + str(totCount) + ' = ' + str(totError/totCount) + '%')

        # TO DO now visualize with tSNE all the zebra embeddings - each a different color
        feat_embedded = TSNE(n_components=2, n_iter=300).fit_transform(allEmbeds)
        print(feat_embedded.shape)
        # plot, color coded so one zebra name has one color


        # visualize 6 errors
        f = plt.figure(figsize=(6, 5))
        for i in range(6):  # plot nine errors
            triplet = wrong_trip[i]
            #sample = sample.byte()
            # plot
            anc = triplet[0].permute(1,2,0)
            print(anc)
            print(type(anc))
            print(anc.shape)
            #anc = transforms.functional.to_pil_image(anc.byte())
            anc = np.asarray(anc)
            pos = triplet[1].permute(1,2,0)
            #pos = transforms.functional.to_pil_image(pos.byte())
            pos = np.asarray(pos)
            neg = triplet[2].permute(1,2,0)
            #neg = transforms.functional.to_pil_image(neg.byte())
            neg = np.asarray(neg)
            ax = plt.subplot(6, 3, i*3 + 1)
            plt.imshow(anc)
            ax.set_title('anchor')
            plt.xticks([])
            plt.yticks([])

            ax = plt.subplot(6, 3, i*3 + 2)
            plt.imshow(pos)
            ax.set_title('pos dist=' + str(wrong_posDis[i]))
            plt.xticks([])
            plt.yticks([])

            ax = plt.subplot(6, 3, i*3 + 3)
            plt.imshow(neg)
            ax.set_title('neg dist=' + str(wrong_negDis[i]))
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()


        return

    # Pretrained torchvision models need specific normalization;
    # see https://pytorch.org/vision/stable/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize([512, 768]), # Some images are slightly different sizes
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    # Initialize dataset loaders
    train_loader = data_loader.get_loader(
        args.data_folder,
        args.train_json,
        transforms,
        batch_size=args.batch_size,
        shuffle=True,
        num_triplets=args.num_train_triplets,
        apply_mask=use_seg,
    )
    val_loader = data_loader.get_loader(
        args.data_folder,
        args.val_json,
        transforms,
        batch_size=args.batch_size,
        shuffle=True,
        num_triplets=int(0.15 * args.num_train_triplets),
        apply_mask=use_seg,
    )

    # object recognition, pretrained on imagenet
    # https://pytorch.org/hub/pytorch_vision_densenet/
    model = initialize_model(use_pretrained=True, l1Units = 500, l2Units=128)
    # print(model)
    model = model.to(device)
    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    trainLoss = []
    valLoss = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch) # None placeholder for triplet loss argument
        trloss = test(model, device, train_loader, "train data") # training loss
        vloss = test(model, device, val_loader, "val data") # validation loss
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
