import torch
import torch.nn as nn
import torch.nn.functional as F

'''
modified to fit dataset size
'''
NUM_CLASSES = 10
freqpts = 64 #freq pts per band

class AlexNet1D(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet1D, self).__init__()
        self.classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # print('convolution input',x.size())
        x = self.features(x.view(-1,1, self.classes*64))
        # print('convolution output',x.size())
        x = x.view(x.size(0), 256 * 2)
        x = self.classifier(x)
        return x

class AlexNet1DsmallFC(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet1DsmallFC, self).__init__()
        self.classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # print('convolution input',x.size())
        x = self.features(x.view(-1,1, self.classes*64))
        # print('convolution output',x.size())
        x = x.view(x.size(0), 256 * 2)
        x = self.classifier(x)
        return x


class AlexNet1DConv(nn.Module):
  #Conv-only version of AlexNet1D
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet1DConv, self).__init__()
        self.classes = num_classes
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2, num_classes),
        )

    def forward(self, x):
        # print('convolution input',x.size())
        x = self.features(x.view(-1,1, self.classes*64))
        # print('convolution output',x.size())
        x = x.view(x.size(0), 256 * 2)
        x = self.classifier(x)
        return x


class TutorialCNN(nn.Module):
#'''tutorial CNN for PSD'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(TutorialCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5, padding = 2)
        self.conv2 = nn.Conv1d(32, 64, 5, padding = 2)
        self.fc1 = nn.Linear(64 *160, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x.view(-1,1,self.classes*freqpts))), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class TutorialCNN2(nn.Module):
#'''smaller conv layers of tutorial CNN for PSD'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(TutorialCNN2, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding = 2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding = 2)
        self.fc1 = nn.Linear(32 *160, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x.view(-1,1,self.classes*freqpts))), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

class TutorialCNN3(nn.Module): 
#'''Conv-only, tutorial CNN'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(TutorialCNN3, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding = 2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding = 2)
        self.fc1 = nn.Linear(32 *160, num_classes)  # 5*5 from image dimension
        self.classes = num_classes

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x.view(-1,1,self.classes*freqpts))), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        return x


class TutorialCNN4(nn.Module):
#'''Larger Conv (64, 128), conv-only tutorial CNN'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(TutorialCNN4, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding = 2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding = 2)
        self.fc1 = nn.Linear(128 *160, num_classes)  # 5*5 from image dimension
        self.classes = num_classes

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x.view(-1,1,self.classes*freqpts))), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        return x

class TutorialCNN5(nn.Module):
#'''Larger Conv (64, 128), conv-only tutorial CNN'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(TutorialCNN5, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5, padding = 2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding = 2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding = 2)
        self.fc1 = nn.Linear(256 *80, num_classes)  
        self.classes = num_classes

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x.view(-1,1,self.classes*freqpts))), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2)
        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        return x


class SmallMLP(nn.Module):
#'''Small MLP'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(640, 1280)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 320)
        self.fc4 = nn.Linear(320, 160)
        self.fc5 = nn.Linear(160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = self.fc4(F.dropout(F.relu(self.fc3(F.dropout(F.relu(self.fc2(F.dropout(F.relu(self.fc1(x.view(-1, self.classes*64)))))))))))
        x = self.fc5(F.dropout(F.relu(x)))
        return x

class BiggerMLP1(nn.Module):
#'''Bigger MLP'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(BiggerMLP1, self).__init__()
        self.classes = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(640, 1280),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1280, 2560),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2560, 2560),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2560, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(640, 320),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.classifier( x.view(-1, self.classes*64))
        return x

class DeepSense(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSense, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(32*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x

class DeepSenseHalf(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseHalf, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(16*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x


class DeepSenseQuarter(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseQuarter, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(4, 8, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(8, 8, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(8*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x

class DeepSenseEighth(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseEighth, self).__init__()
        self.conv1 = nn.Conv1d(1, 2, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(2, 4, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(4, 4, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(4*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x


class DeepSenseSmall(nn.Module):
#'''DeepSense to be finished'''
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepSenseSmall, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding = 2)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding = 2)
        self.conv3 = nn.Conv1d(1, 2, kernel_size=5, stride=1, padding = 2)
        self.conv4 = nn.Conv1d(2, 2, kernel_size=5, stride=1, padding = 2)
        self.fc3 = nn.Linear(2*160, num_classes)
        self.classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x.view(-1,1,self.classes*freqpts)))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)

        x = F.dropout(torch.flatten(x, 1)) # flatten all dimensions except the batch dimension
        x = self.fc3(x)
        return x