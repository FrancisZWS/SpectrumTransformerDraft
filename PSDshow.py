import matplotlib.pyplot as plt
import numpy as np

def imshow(img, max_val):
    #img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    npimg = img
    #plt.imshow(np.transpose(npimg, cmap = 'gray', interpolation='nearest' ))
    plt.rcParams['figure.figsize'] = (12., 8.)
    plt.imshow(-1*img , cmap = 'gray', vmin = -1*max_val, vmax = 0.0, interpolation='nearest'   )# notice its negative
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.axis('off')
    plt.show()