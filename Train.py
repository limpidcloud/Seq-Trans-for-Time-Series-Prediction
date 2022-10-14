import time
import numpy as np
import torch.utils.data as data

from Structure import SeqTrans
from config import *

length = str(predict_len)
torch.cuda.empty_cache()
print('Model: Seq-Trans')
print('Dataset:', data_name, ' ', 'Predict Length:', length)

print('----Loading Dataset----')
x_train = torch.load(data_name + 'Data/x_train.pth').to(device)
y_train = torch.load(data_name + 'Data/y_train.pth').to(device)
dataset = data.TensorDataset(x_train, y_train)
train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('--------Success!-------')
print('{l_(i)}:', seq_len_list)
print('{n_(i)}:', tmp_list)
print('d_conv:', conv_num)
print('in_channel:', in_dim, ' out_channel:', out_dim)

print('----Training Starts----')
print('Epoch:', epoch)
model = SeqTrans(dropout=0.2).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
time_list = []
loss_list = []
for i in range(epoch):
    start_time = time.time()
    avg_loss = 0
    for j, (v, l) in enumerate(train_loader):
        optimizer.zero_grad()
        x = v[:, :, :in_dim]
        x_phase = v[:, :, in_dim:]
        y = v[:, -predict_len:, :in_dim]  # decoder input is historical sequence with length l_pred
        y_phase = l[:, :, in_dim:]
        label = l[:, :, in_dim - out_dim:in_dim]  # the last out_dim attributes are the sequence to predict
        predict = model(x, x_phase, y, y_phase)
        loss = criterion(predict, label)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
    end_time = time.time()
    print('epoch:', i, ' time:', round(end_time - start_time, 2), ' Sum_MSE:', avg_loss)
    time_list.append(round(end_time - start_time, 2))
    loss_list.append(avg_loss)

torch.save(model, 'Results/Seq-Trans.pth')
time_list = np.array(time_list)
loss_list = np.array(loss_list)
np.savetxt('Results/time.txt', time_list)
np.savetxt('Results/loss.txt', loss_list)
