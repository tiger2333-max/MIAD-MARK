from utils import MyDataset, show_confMat, validate
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
from configs import args
from loss import FocalLoss
from dataset import get_mean_std
from utils import get_model
from dataset import classname_dict

if __name__=='__main__':

    root_path = args.root_path
    model_name = args.model_name
    dataset = args.dataset
    train_bs = args.bs
    valid_bs = args.bs
    lr_init = args.lr_init
    cla_num = args.cla_num
    max_epoch = args.max_epoch

    class_name = classname_dict[dataset]
    print(class_name)

    # log
    result_dir = os.path.join(root_path, 'data', dataset, 'result', model_name)
    # print(result_dir)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')

    log_dir = os.path.join(result_dir, time_str)
    if not os.path.join(result_dir, time_str):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # Config Transforms

    if model_name == 'inceptionv3':
        size = 299
    else:
        size = 224

    mean,std = get_mean_std(args)

    normTransform = transforms.Normalize(mean,std)
    trainTransform = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ColorJitter(brightness=0.5, contrast=(0.2, 0.8)),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normTransform
    ])

    validTransform = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor(),
        normTransform
    ])

    #txt
    train_path = os.path.join(root_path, 'data', dataset, 'train.txt')
    valid_path = os.path.join(root_path, 'data', dataset, 'test.txt')

    # MyDataset
    train_data = MyDataset(txt_path=train_path, transform=trainTransform)
    valid_data = MyDataset(txt_path=valid_path, transform=validTransform)

    # DataLoder
    train_loader = DataLoader(train_data, batch_size=train_bs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=valid_bs)

    # check gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda is available:', torch.cuda.is_available())
    print('gpu numbers:', torch.cuda.device_count())
    print('cuda version:', torch.version.cuda)

    # load model
    net = get_model(model_name, cla_num, device, pretrained=True)
    print(net.named_modules)

    # define loss, opt, lr
    criterion = None
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'Focal':
        criterion = FocalLoss(num_class=args.cla_num)

    optimizer = None
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # lr decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

    # print('loss',args.loss,'num class',args.cla_num)

    # ------------------------------------ Train ------------------------------------

    for epoch in range(max_epoch):

        loss_sigma = 0.0  # sum of the loss in one epoch
        correct = 0.0
        total = 0.0

        for i, data in enumerate(train_loader):

            inputs, labels = data[0].to(device), data[1].to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            # forward, backward, update weights
            optimizer.zero_grad()

            outputs = net(inputs)
            if model_name == 'vit':
                outputs = net(inputs).logits
                print(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            scheduler.step()  # update lr

            # Statistic predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()
            loss_sigma += loss.item()

            # Print training progress every 10 epochs，loss is average of 10 epochs
            if i % 10 == 9:
                loss_avg = loss_sigma / 10
                loss_sigma = 0.0
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # record loss, lr and acc
                writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)

                writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)

                writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

        # record grad and weight
        for name, layer in net.named_parameters():
            if args.model_name == 'inceptionv3':
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
            else:
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

        # ------------------------------------ Validation ------------------------------------

        if epoch % 2 == 0:
            loss_sigma = 0.0
            cls_num = len(class_name)
            conf_mat = np.zeros([cls_num, cls_num])  # Confusion Matrix
            net.eval()
            for i, data in enumerate(valid_loader):

                images, labels = data[0].to(device), data[1].to(device)
                images, labels = Variable(images), Variable(labels)

                # forward
                outputs = net(images)
                if model_name == 'vit':
                    outputs = net(images).logits
                outputs.detach_()

                # loss
                loss = criterion(outputs, labels)
                loss_sigma += loss.item()


                _, predicted = torch.max(outputs.data, 1)
                # labels = labels.data    # Variable --> tensor

                #
                for j in range(len(labels)):
                    cate_i = labels[j].cpu().numpy()
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i, pre_i] += 1.0

            print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))

            writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(valid_loader)}, epoch)
            writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
    print('Finished Training')

    # ------------------------------------ Save Model and ConfMat ------------------------------------

    net_save_path = os.path.join(log_dir, model_name + '.pkl')
    torch.save(net.state_dict(), net_save_path)

    conf_mat_train, train_acc = validate(net, train_loader, 'train', class_name, model_name, device)
    conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', class_name, model_name, device)

    show_confMat(conf_mat_train, class_name, 'train', log_dir)
    show_confMat(conf_mat_valid, class_name, 'valid', log_dir)

    writer.close()
