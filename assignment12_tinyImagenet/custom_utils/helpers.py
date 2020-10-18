
import time

import numpy as np
import torch
from tqdm import tqdm



def test(model, device, test_loader, loss_func):

    if (device.type == "cuda") and (not next(model.parameters()).is_cuda):
        model = model.to(device)

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, correct / len(test_loader.dataset)



def train(model, device, train_loader, optimizer, epoch, loss_func):

    if (device.type == "cuda") and (not next(model.parameters()).is_cuda):
        model = model.to(device)

    model.train()

    train_loss = 0
    correct = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
    
    total_imgs = len(train_loader.dataset)
    train_loss /= total_imgs
    accuracy = 100. * correct / total_imgs

    print('\nTrain Data: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        train_loss, correct, total_imgs, accuracy)
    )
    
    return train_loss, accuracy



def record_max_acc(max_acc, accuracy_store_path):
    f = open(accuracy_store_path, "w")
    f.write(str(max_acc))
    f.close()



def one_cycle_lr(epoch_num, peak_epoch, last_epoch, model, max_lr, min_lr=None, optim_class=torch.optim.SGD):

    if not min_lr:
        min_lr = max_lr / 10.0

    a = np.linspace(min_lr, max_lr, peak_epoch)
    b = np.linspace(max_lr, min_lr, last_epoch-peak_epoch+1)

    lrs = np.concatenate([a, b[1:]])

    lr = lrs[epoch_num-1]

    optimizer = optim_class(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)

    return optimizer



def train_epochs(model, device, train_loader, test_loader, 
                 optimizer, loss_func, epochs,
                 accuracy_store_path=None, model_sd_save_path=None,
                 save_if_better_acc=False,
                 manual_scheduler=None, auto_scheduler=None,
                 oclr_params=None):

    if (bool(save_if_better_acc) + bool(model_sd_save_path)) == 1:
        raise Exception("If save_if_better_acc is True, then "
                        "model_sd_save_path must be given! "
                        "Similary, if model_sd_save_path is given, then "
                        "save_if_better_acc must be True!")

    if accuracy_store_path:
        try:
            with open(accuracy_store_path, "r") as infl:
                max_acc = float(infl.read().strip())
        except:
            max_acc = 0.0
        print("\nLAST RECORDED MAX ACCURACY: ", max_acc)

    
    train_acc = []
    test_acc = []
    train_losses = []
    test_losses = []

    start = time.time()
    for epoch in range(1, epochs+1):

        if oclr_params:
            peak_epoch = oclr_params["peak_epoch"]
            last_epoch = epochs+1
            max_lr = oclr_params["max_lr"]
            min_lr = oclr_params.get("min_lr", None)
            optimizer = one_cycle_lr(epoch_num=epoch, peak_epoch=peak_epoch, 
                                    last_epoch=last_epoch, model=model, max_lr=max_lr,
                                    min_lr=min_lr)

        ep_start = time.time()
        print("================================================================")
        print("EPOCH NUM {},  LR USED: {}".format(epoch, 
                                optimizer.state_dict()["param_groups"][0]["lr"]))
        
        trl, tra = train(model=model, 
                        device=device, 
                        train_loader=train_loader, 
                        optimizer=optimizer, 
                        epoch=epoch, 
                        loss_func=loss_func)

        if manual_scheduler:
            manual_scheduler.step()

        tsl, tsa = test(model=model, 
                        device=device, 
                        test_loader=test_loader,
                        loss_func=loss_func)

        if auto_scheduler:
            auto_scheduler.step(tsa)
        
        train_acc.append(tra)
        test_acc.append(tsa)
        train_losses.append(trl)
        test_losses.append(tsl)

        if save_if_better_acc and model_sd_save_path:
            if tsa > max_acc:
                max_acc = tsa
                torch.save(model.state_dict(), model_sd_save_path)
                record_max_acc(max_acc=max_acc, accuracy_store_path=accuracy_store_path)
        print("-----------------------------------------------")
    print("TOTAL TRAINING TIME: ", time.time() - start)
    print("LAST 10 EPOCH AVG ACC: ", sum(test_acc[-10:]) / len(test_acc[-10:]) )
    print("LAST 5 EPOCH AVG ACC: ", sum(test_acc[-5:]) / len(test_acc[-5:]) )
    print("MAX ACCURACY: ", max(test_acc))

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_acc,
        "test_accuracies": test_acc
    }



def add_l1_reg(func, model, lambda_l1):

    def inner(output, target, *args, **kwargs):

        l1 = 0
        loss = func(output, target, *args, **kwargs)
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        return loss + (lambda_l1 * l1)

    return inner



def test_lr_effectiveness(model, device, train_loader, test_loader, 
                          optimizer, loss_func, iterations):
    """
    """
    if (device.type == "cuda") and (not next(model.parameters()).is_cuda):
        model = model.to(device)

    model.train()

    dl_iterator = iter(train_loader)

    train_loss = 0
    correct = 0
    pbar = tqdm(range(iterations), position=0, leave=True)

    for inum in pbar:

        try:
            data, target = next(dl_iterator)
        except StopIteration:
            dl_iterator = iter(train_loader)
            data, target = next(dl_iterator)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        t_loss = loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        crrct = pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()


        train_loss += tloss
        correct += crrct
        pbar.set_description(desc= f'loss={tloss} iteration={inum}')
    
    total_imgs = len(train_loader.dataset)
    train_loss /= total_imgs
    accuracy = 100. * correct / total_imgs

    # print('\nTrain Data: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
    #     train_loss, correct, total_imgs, accuracy)
    # )


    tst_loss, tst_acc = test(model=model, device=device, test_loader=test_loader, loss_func=loss_func)

    print('\nTest Loss: {:.4f}, Accuracy: {:.4f}%'.format(tst_loss, tst_acc))
    
    return tst_loss, tst_acc



def run_lr_range_test(model_class, device, train_loader, test_loader,
                      loss_func, iterations, lr_start, lr_end):
    lr_range = [lr_start]
    lr = lr_start / 0.1
    while lr <= lr_end:
        lr_range.append(lr)
        lr = lr/0.1

    yaxis = []

    for lr in lr_range:
        model = model_class()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0, momentum=0.9)

        tst_loss, tst_acc = test_lr_effectiveness(
                                 model=model, device=device, train_loader=train_loader,
                                 test_loader=test_loader, optimizer=optimizer,
                                 loss_func=loss_func, iterations=iterations)

        print(f'lr={lr}, test_accuracy={tst_acc}')

        yaxis.append(tst_acc)

    return lr_range, yaxis


