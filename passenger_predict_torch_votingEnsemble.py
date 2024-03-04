import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from preprocessing import load_bus, load_delay, load_passenger, load_weather
from function_package import split_xy
from sklearn.metrics import r2_score
from torcheval.metrics import R2Score
import copy
from skorch import NeuralNetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
'''
승하차 합산 승객 변동량 예측모델
'''
print(torch.__version__)    # 2.2.0+cu118

# 변수 설정
device = (
    # "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)

# data
def data_gen(station_num):
    # 데이터 로드
    bus_csv = load_bus()
    passenger_csv, passenger_scaler = load_passenger(return_scaler=True)
    weather_csv = load_weather()
    
    station_list = list(passenger_csv.columns)
    # print(bus_csv.shape,weather_csv.shape,passenger_csv.shape)  # (5832, 2) (5832, 3) (5832, 282)
    
    df = pd.DataFrame(np.concatenate((bus_csv, weather_csv, passenger_csv,), axis=1))
    # print(df.shape) # (5832, 287)
    data = pd.DataFrame()
    y = pd.DataFrame()
    data = df.iloc[:-1,:]
    y = df.iloc[1:,station_list.index(station_num)+5] # 5부터 역명
 
    x_train, x_test, y_train, y_test = train_test_split(
        data, y, train_size=0.9, random_state=100)
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return data, x_train, y_train, x_test, y_test, passenger_scaler

# model  
class TorchDNN(nn.Module):
    def __init__(self,input_shape,output_shape) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape,256,device=device),
            nn.ReLU(),
            nn.Linear(256,128,device=device),
            nn.ReLU(),
            nn.Linear(128,output_shape,device=device)
        )
        
    def forward(self, x=None, *args, **x_dict):
        print("x:",x,"\nargs: ", args,"\nx_dict: ", x_dict)
        if type(x) == None and x_dict:
            print("x is dict")
            x_dict_old_keys = list(x_dict.keys())
            x_dict_new_keys = np.array(x_dict_old_keys).astype(str)
            for old_key, new_key in zip(x_dict_old_keys,x_dict_new_keys):
                x_dict[new_key] = x_dict.pop(old_key)
            x = x_dict
        logits = self.linear_relu_stack(x)
        logits = logits.reshape(-1,)
        return logits
    
def passenger_predict(station_num)->tuple[np.ndarray, float, float]:
    data, x_train, y_train, x_test, y_test, delay_scaler = data_gen(station_num)
    
    xgb_params = {'learning_rate': 0.13349839953884737,
                'n_estimators': 99,
                'max_depth': 8,
                'min_child_weight': 3.471164143831403e-06,
                'subsample': 0.6661302167437514,
                'colsample_bytree': 0.9856906281904222,
                'gamma': 4.5485144879936555e-06,
                'reg_alpha': 0.014276113125688179,
                'reg_lambda': 10.121476098960851}

    cat_params = {'iterations': 258,
                'learning_rate': 0.10372129288289585,
                'depth': 4,
                'l2_leaf_reg': 7.502013869614672,
                'bagging_temperature': 0.0006871304265196931,
                'random_strength': 0.002522427032110104,
                'border_count': 348}

    lgbm_params = {'learning_rate': 0.27047557712428816,
                'n_estimators': 84,
                'num_leaves': 18,
                'max_depth': 6,
                'min_child_samples': 19,
                'subsample': 0.6791274154754994,
                'colsample_bytree': 0.8780923690420561,
                'reg_alpha': 0.030054587470444684,
                'reg_lambda': 5.627270968507918,
                'min_split_gain': 2.7094312530442e-07,
                'min_child_weight': 8.308879346847586e-06,
                'cat_smooth': 59}

    my_dnn = NeuralNetRegressor(TorchDNN(input_shape=x_train.shape[1],output_shape=1),
                            max_epochs=1000,
                            device=device,
                            criterion=nn.MSELoss,
                            optimizer=torch.optim.Adam,
                            )

    model = VotingRegressor([
        # ('My_DNN',my_dnn),
        # ('MyLSTM',my_lstm),
        ('RandomForestRegressor',RandomForestRegressor()),
        ('XGBRegressor',XGBRegressor(**xgb_params)),
        # ('CatBoostRegressor',CatBoostRegressor(**cat_params)), # error
        # ('AdaBoostRegressor',AdaBoostRegressor()),
        ('LGBMRegressor',LGBMRegressor(**lgbm_params)),
        # ('SVR',SVR()),
        ('LinearRegression',LinearRegression()),
    ])
    
    import os.path
    PATH = f'./model_save/passenger_predict/'
    if os.path.exists(PATH+f'passenger_ensemble_{station_num}.pkl'):
        model = pickle.load(open(PATH+f'passenger_ensemble_{station_num}.pkl', 'rb'))
    else:
    # fit & eval
        model.fit(x_train,y_train)
    # model.fit(x_train,y_train) # 임시
    
    r2 = model.score(x_test,y_test)
    y_predict = model.predict(x_test)
    y_submit = model.predict(data)
    loss = mean_squared_error(y_predict,y_test)
    print("R2:   ",r2)
    print("LOSS: ",loss)

    # 결과를 파일로 저장해서 확인
    y_submit = delay_scaler.inverse_transform(y_submit.reshape(-1,1))
    y_submit = y_submit.reshape(-1)

    # 모델 저장
    pickle.dump(model,open(PATH+f'passenger_ensemble_{station_num}.pkl', 'wb'))
    return y_submit, r2, loss

if __name__ == '__main__':
    passenger_csv = load_passenger()
    station_list = passenger_csv.columns
    print(station_list)
    r2_list = []
    for n, station_num in enumerate(station_list):
        result = passenger_predict(station_num)
        print(n+1,'/282')
        r2_list.append(result[1])
    print(r2_list)
    print("DONE")
    


# only my_dnn

# only RandomForestRegressor
# R2:    0.9481817800630611
# LOSS:  1.75546211929742e-05

# only XGBRegressor
# R2:    0.9486495236740224
# LOSS:  1.7396162220129257e-05

# only LGBMRegressor
# R2:    0.9487035226063297
# LOSS:  1.7377868832154276e-05

# only CatBoostRegressor
# R2:    0.9445804646910017
# LOSS:  1.8774650117739953e-05

# ensemble cat, xgb, RF
# R2:    0.9490893959460077
# LOSS:  1.724714530836748e-05

# ensemble cat, xgb, RF, LGBM 
# R2:    0.9508556422551188
# LOSS:  1.6648788496271948e-05

# ensemble cat, xgb, RF, LGBM, LinearRegressor
# R2:    0.951682501080484
# LOSS:  1.6368670933819583e-05

# ensemble cat, xgb, RF, LGBM, LinearRegressor, SVR
# R2:    0.9187888084290099
# LOSS:  2.7512170553016626e-05
