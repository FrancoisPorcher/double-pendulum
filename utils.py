from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def min_max_scaler():
    long_trajectory = pd.read_csv('data/trajectory.csv').values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(long_trajectory.reshape(-1, 4))
    return scaler



def prediction(model, trajectory, lookback = 40):
    scaler = min_max_scaler()
    output_dim = model.output_size
    
    initial_state = trajectory[:lookback]
    initial_state = scaler.transform(initial_state)
    
    number_of_steps = trajectory.shape[0] - lookback
    
    hn, cn = model.init(batch_size=1)
    model.eval()
    x = torch.from_numpy(initial_state).float().view(-1, 4).to(device)  
    final_trajectory = x.cpu().detach().numpy()
    

    for i in range(number_of_steps):
        out , hn , cn = model(x.reshape(lookback, 1, output_dim),hn,cn)
        final_trajectory = np.append(final_trajectory, out.cpu().detach().numpy(), axis=0)
        x = torch.cat((x[1:], out))
        
    final_trajectory = scaler.inverse_transform(final_trajectory)
    
    return final_trajectory
    