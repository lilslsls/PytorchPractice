import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False
if_use_gpu = 1

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,
    transform = torchvision.transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
    
)

print(train_data.data.size())
print(train_data.targets.size())
# plt.imshow(train_data.data[1].numpy(),cmap='gray')
# plt.title('%i'%train_data.targets[1])
# plt.show()

train_loader = Data.DataLoader(dataset = train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x = torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255.
test_x = test_x.cuda()
test_y = test_data.targets[:2000]
test_y = test_y.cuda()
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding = 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output,x


cnn = CNN()
cnn = cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = cnn(b_x)[0]
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output,last_layer = cnn(test_x)
            pred_y = torch.max(test_output,1)[1].data.cpu().numpy()
            print(test_y.size())
            accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.size(0))
            print('Epoch:',epoch,'|train loss:%.4f'%loss,'|test accuracy:%.2f'%accuracy)
            
