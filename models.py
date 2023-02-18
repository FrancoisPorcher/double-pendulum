
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

    


class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size , num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_dim
        self.output_size = output_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_dim , hidden_size = hidden_size , num_layers= num_layers)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self,x,hn,cn):
        out , (hn,cn) = self.lstm(x , (hn,cn))
        final_out = self.fc(out[-1])
        return final_out,hn,cn

    def predict(self,x):
        hn,cn  = self.init()
        final_out = self.fc(out[-1])
        return final_out

    def init(self, batch_size):
        h0 =  torch.zeros(self.num_layers , batch_size , self.hidden_size).to(device)
        c0 =  torch.zeros(self.num_layers , batch_size , self.hidden_size).to(device)
        return h0 , c0





    
    

#this is the full OneStep function.   it 
def OneStep(double_pendulum, trajectory, model):
    """_summary_

    Args:
        double_pendulum (class): Double pendulum class
        trajectory (_type_): trajectory computed by the finite difference method
        model (_type_): lstm model
        steps (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    
    # Get the parameters from the double pendulum
    L1 = double_pendulum.l1
    L2 = double_pendulum.l2
    
    print('trajectory set length =', len(trajectory))
    train_window = 10
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_trajectory_normalized = scaler.fit_transform(trajectory.reshape(-1, 4))
    fut_pred = len(trajectory) - train_window
    test_inputs = train_trajectory_normalized[0:train_window].reshape(train_window,4).tolist()
    #print(test_inputs)
    s2 = train_trajectory_normalized.reshape(len(trajectory),4).tolist()
    realtrajectory = trajectory
    
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
    u0 = trajectory[:,0]     # theta_1 
    u1 = trajectory[:,1]     # omega 1
    u2 = trajectory[:,2]     # theta_2 
    u3 = trajectory[:,3]     # omega_2 
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
    plt.plot(x2[:], y2[:], color='r')
    plt.plot(xp2[:],yp2[:] , color='g')
    err = 0.0
    errs = 0.0
    cnt = 0
    badday  =0.0
    errsq = 0.0
    maxerr = 0.0
    maxloc = 0
    
    #the following attempts to make an estimate of the error.  not very good.
    for i in range(len(actual_predictions)):
        er =np.linalg.norm(realtrajectory[i]-actual_predictions[i])
        err += er
        if er > maxerr:
            maxerr = er
            maxloc = i
    print("mean error =", err/101)
    print('maxerr =', maxerr, ' at ', maxloc)
    return actual_predictions