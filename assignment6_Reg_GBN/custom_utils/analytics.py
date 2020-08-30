
import matplotlib.pyplot as plt



def create_plot_pos(nrows, ncols):
    num_images = nrows * ncols
    positions = []
    for r in range(num_images):
        row = r // ncols
        col = r % ncols
        positions.append((row, col))
    return positions



def plot_misclassified(imgs, targets, preds, nrows, ncols, skip=0, 
                       plt_scaler=(2,2.5), plt_fsize=12):
    """
    imgs is a tensor of all images
    targets is a tensor of all label targets
    preds is a tensor of all predictions from the model
    """
    matches = preds.eq(targets)

    total_imgs = nrows*ncols
    pos = create_plot_pos(nrows, ncols)

    fig, axes = plt.subplots(nrows=nrows,
                             ncols=ncols, 
                             figsize=(ncols*plt_scaler[1], nrows*plt_scaler[0]), 
                             sharex=True, 
                             sharey=True)

    idx = 0
    posidx = 0
    total_skipped = 0
    for m in matches:
        if posidx > total_imgs-1:
            break

        if not m:
            if total_skipped <= skip:
                total_skipped += 1
                idx += 1
                continue

            img = imgs[idx].reshape(28,28)
            title = "Act: " + str(targets[idx].item()) + ", Pred: " + str(preds[idx].item())
            chart_pos = pos[posidx]
            axes[chart_pos].imshow(img)
            axes[chart_pos].set_title(title, fontsize=plt_fsize)
            axes[chart_pos].axis("off")

            posidx += 1
        
        idx += 1
    
    return fig, axes

