import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

class MyLSTM(nn.Module):
    def __init__(self, hidden_size = 100):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden_size)
        self.linear = nn.Linear(hidden_size, 4)
        self.c_h = (torch.zeros(1,1,hidden_size),
                    torch.zeros(1,1,hidden_size))
    def forward(self, x):
        h, self.c_h= self.lstm(x.view(len(x) ,1, -1), self.c_h)
        predictions = self.linear(h.view(len(x), -1))
        return predictions[-1]

#this is the full OneStep function.   it 
def OneStep(double_pendulum, data, model, steps = 100):
    L1 = double_pendulum.l1
    L2 = double_pendulum.l2
    
    print('data set length =', len(data))
    train_window = 10
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(data.reshape(-1, 4))
    fut_pred = len(data) - train_window
    test_inputs = train_data_normalized[0:train_window].reshape(train_window,4).tolist()
    #print(test_inputs)
    s2 = train_data_normalized.reshape(len(data),4).tolist()
    realdata = data
    
    model.eval()
    preds = test_inputs.copy()
    t2 = test_inputs
    hidden_layer_size = 100
    x = 0
    for i in range(fut_pred):
        seq = torch.FloatTensor(t2[i:])
        model.c_h = (torch.zeros(1, 1, hidden_layer_size),
                        torch.zeros(1, 1, hidden_layer_size))
        x = model(seq)
        preds.append(x.detach().numpy())
        t2.append(x.detach().numpy())
    actual_predictions = scaler.inverse_transform(np.array(preds ).reshape(-1,4))
    print(len(actual_predictions))
    
    # the following will plot the lower mass path for steps using the actual ODE sover
    # and the predicitons
    plt.figure( figsize=(10,5))
    u0 = data[:,0]     # theta_1 
    u1 = data[:,1]     # omega 1
    u2 = data[:,2]     # theta_2 
    u3 = data[:,3]     # omega_2 
    up0 = actual_predictions[:,0]     # theta_1 
    up1 = actual_predictions[:,1]     # omega 1
    up2 = actual_predictions[:,2]     # theta_2 
    up3 = actual_predictions[:,3]     # omega_2 
    x1 = L1*np.sin(u0);          # First Pendulum
    y1 = -L1*np.cos(u0);
    x2 = x1 + L2*np.sin(u2);     # Second Pendulum
    y2 = y1 - L2*np.cos(u2);
    xp1 = L1*np.sin(up0);          # First Pendulum
    yp1 = -L1*np.cos(up0);
    xp2 = xp1 + L2*np.sin(up2);     # Second Pendulum
    yp2 = yp1 - L2*np.cos(up2);
    print(x2[0], y2[0])
    plt.plot(x2[0:steps], y2[0:steps], color='r')
    plt.plot(xp2[0:steps],yp2[0:steps] , color='g')
    err = 0.0
    errs = 0.0
    cnt = 0
    badday  =0.0
    errsq = 0.0
    maxerr = 0.0
    maxloc = 0
    
    #the following attempts to make an estimate of the error.  not very good.
    for i in range(len(actual_predictions)):
        er =np.linalg.norm(realdata[i]-actual_predictions[i])
        err += er
        if er > maxerr:
            maxerr = er
            maxloc = i
    print("mean error =", err/101)
    print('maxerr =', maxerr, ' at ', maxloc)
    return actual_predictions