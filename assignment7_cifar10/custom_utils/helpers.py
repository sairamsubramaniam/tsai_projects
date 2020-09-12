
import time

import torch
from tqdm import tqdm



def test(model, device, test_loader, loss_func):

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



def train_epochs(model, device, train_loader, test_loader, optimizer, loss_func, epochs,
                 accuracy_store_path=None, model_sd_save_path=None,
                 save_if_better_acc=False):

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
    for epoch in range(1, epochs):

        ep_start = time.time()
        print()
        print("EPOCH NUM {}".format(epoch))
        
        trl, tra = train(model=model, 
                        device=device, 
                        train_loader=train_loader, 
                        optimizer=optimizer, 
                        epoch=epoch, 
                        loss_func=loss_func)
        tsl, tsa = test(model=model, 
                        device=device, 
                        test_loader=test_loader,
                        loss_func=loss_func)
        
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


