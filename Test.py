from config import *
import matplotlib.pyplot as plt

length = str(predict_len)
model = torch.load('Results/Seq-Trans.pth')
x_test = torch.load(data_name + 'Data/x_test.pth').to(device)
y_test = torch.load(data_name + 'Data/y_test.pth').to(device)
value_en = x_test[:, :, :in_dim]
phase_en = x_test[:, :, in_dim:]
value_de = x_test[:, -predict_len:, :in_dim]
phase_de = y_test[:, :, in_dim:]
label = y_test[:, :, in_dim - out_dim:in_dim]

criterion = torch.nn.MSELoss()
avg_loss = 0
avg_loss2 = 0
pieces = value_en.shape[0]
for i in range(0, pieces, batch_size):
    predict = model(value_en[i:i + batch_size], phase_en[i:i + batch_size], value_de[i:i + batch_size], phase_de[i:i + batch_size])
    # predict = predict - predict[:,:1,:].repeat(1,predict_len,1)+value_de[i:i+batch_size,-1:,in_dim - out_dim:in_dim].repeat(1,predict_len,1)
    # the above process in 20th line is explained in 'ReadMe.txt'.
    loss = criterion(predict, label[i:i + batch_size])
    loss2 = torch.mean(torch.abs(predict - label[i:i + batch_size]))
    avg_loss += loss.item()
    avg_loss2 += loss2.item()
print('MSE', avg_loss / len(range(0, pieces, batch_size)))
print('MAE', avg_loss2 / len(range(0, pieces, batch_size)))

index = 0  # chose one test sequence
channel = 0  # choose one attribute to show
pred = model(value_en[index:index + 1], phase_en[index:index + 1], value_de[index:index + 1], phase_de[index:index + 1])
pred = pred.cpu().detach().numpy()
label = label[index:index + 1].cpu().detach().numpy()
x = range(predict_len)
plt.figure()
plt.plot(x, pred[0,:,channel], label='Prediction')
plt.plot(x, label[0,:,channel], label='Real')
plt.legend()
plt.savefig('Results/'+data_name+'Prediction.png')
plt.show()