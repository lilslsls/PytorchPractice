import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

EPOCH = 4
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root = './mnist/',
    train = True,
    transform = transforms.ToTensor(),
    download = DOWNLOAD_MNIST,
)
print(train_data.data.size())
train_loader = Data.DataLoader(dataset = train_data,batch_size = BATCH_SIZE,shuffle = True)

test_data = dsets.MNIST(
    root='./mnist/', 
    train=False, 
    transform=transforms.ToTensor()
)
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255.
test_x = test_x.cuda()
test_y = test_data.targets[:2000]
test_y = test_y.cuda()

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(
            input_size = INPUT_SIZE,
            hidden_size = 64,
            num_layers = 2,
            batch_first = True
        )

        self.out = nn.Linear(64,10)
    def forward(self, x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
rnn = rnn.cuda()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(b_x,b_y) in enumerate(train_loader):
        b_x = b_x.view(-1,28,28)
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output =rnn(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)                   # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
            # accuracy = float((pred_y == test_y.cpu().numpy()).astype(int).sum()) / float(test_y.size)
            accuracy = float((pred_y == test_y.data.cpu().numpy()).astype(int).sum())/float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.2f' % accuracy)
