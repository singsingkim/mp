import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
import lightgbm as lgb
from keras.layers import concatenate
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_deay()

# 레이블 선택
x1 = passenger_csv
x2 = weather_csv
x3 = delay_csv
# y = delay_csv['1호선지연(분)']
# y = delay_csv['2호선지연(분)']
# y = delay_csv['3호선지연(분)']
# y = delay_csv['4호선지연(분)']
# y = delay_csv['5호선지연(분)']
# y = delay_csv['6호선지연(분)']
# y = delay_csv['7호선지연(분)']
y = delay_csv['8호선지연(분)']

# 훈련 및 테스트 데이터 분할(원하는 상황으로 주석처리를 바꾸기)
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1,x2,x3, y, train_size=0.8, random_state=100, stratify=y)

# 스케일링(모든 데이터 이용시)
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
scaler3 = RobustScaler()
# scaler = MaxAbsScaler()
x1_train_scaled = scaler1.fit_transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

x2_train_scaled = scaler2.fit_transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

x3_train_scaled = scaler3.fit_transform(x3_train)
x3_test_scaled = scaler3.transform(x3_test)
# 스케일링(각각)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# 데이터 연결(모든 데이터 이용시)
x_train = np.concatenate((x1_train_scaled, x2_train_scaled, x3_train_scaled), axis=1)
x_test = np.concatenate((x1_test_scaled, x2_test_scaled, x3_test_scaled), axis=1)
s_t = time.time()
import random
def objective(trial):
    rd=random.randint(1,1000)
    params = {
        "metric": "mse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": rd,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 100.0),
        'min_split_gain': trial.suggest_loguniform('min_split_gain', 1e-8, 100.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'cat_smooth': trial.suggest_int('cat_smooth', 1, 100),
        "early_stopping_rounds": 3,  # 얼리 스탑
    }
    
    model = lgb.LGBMRegressor(**params, device='gpu')
    
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              eval_metric='mse',)
    
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    
    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=110)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
e_t = time.time()
 
#데이터 폴더에 다른 파일이 들어가는걸 방지하기위해 경로를 다른곳으로 설정해둔것
model_path = 'c:/_data/_save/project/passenger+weather+bus_lightgbm_8호선.pkl'
params = study.best_params
model = LGBMRegressor(**params, tree_method='gpu_hist')
model.fit(x_train, y_train)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print("Best parameters found: ", study.best_params)
print("r2: ", study.best_value)
print("시간 :", e_t-s_t)

'''
1호선
Best parameters found:  {'learning_rate': 0.27047557712428816, 'n_estimators': 84, 'num_leaves': 18, 'max_depth': 6, 'min_child_samples': 19, 'subsample': 0.6791274154754994, 'colsample_bytree': 0.8780923690420561, 'reg_alpha': 0.030054587470444684, 'reg_lambda': 5.627270968507918, 'min_split_gain': 2.7094312530442e-07, 'min_child_weight': 8.308879346847586e-06, 'cat_smooth': 59}
r2:  0.9999999975583862
시간 : 33.35918116569519
2호선
Best parameters found:  {'learning_rate': 0.14413734559781788, 'n_estimators': 198, 'num_leaves': 42, 'max_depth': 6, 'min_child_samples': 7, 'subsample': 0.9557608969851219, 'colsample_bytree': 0.9838882584209773, 'reg_alpha': 0.1897440307140042, 'reg_lambda': 25.126733008821144, 'min_split_gain': 0.003655976010374016, 'min_child_weight': 0.5922804128641153, 'cat_smooth': 62}
r2:  0.9999927509727476
시간 : 45.06050181388855
3호선
Best parameters found:  {'learning_rate': 0.17008051793938428, 'n_estimators': 130, 'num_leaves': 23, 'max_depth': 3, 'min_child_samples': 10, 'subsample': 0.9318520223176201, 'colsample_bytree': 0.9889688069812849, 'reg_alpha': 0.019986334680938297, 'reg_lambda': 0.04674673773248154, 'min_split_gain': 0.00023699977060134383, 'min_child_weight': 8.493545431078861e-08, 'cat_smooth': 13}
r2:  0.9999999802199471
시간 : 31.26534366607666
4호선
Best parameters found:  {'learning_rate': 0.27194886857762424, 'n_estimators': 60, 'num_leaves': 42, 'max_depth': 8, 'min_child_samples': 12, 'subsample': 0.5533904208459461, 'colsample_bytree': 0.9960932157622057, 'reg_alpha': 0.010969832597643048, 'reg_lambda': 3.2345086555316818, 'min_split_gain': 8.204842748203165e-06, 'min_child_weight': 0.0001781633591036314, 'cat_smooth': 93}
r2:  0.9999999975307808
시간 : 35.21856713294983
5호선
Best parameters found:  {'learning_rate': 0.12518801855182402, 'n_estimators': 71, 'num_leaves': 19, 'max_depth': 4, 'min_child_samples': 6, 'subsample': 0.7190606998366449, 'colsample_bytree': 0.7764466979984042, 'reg_alpha': 0.018551748005393432, 'reg_lambda': 0.012374120406418929, 'min_split_gain': 0.06949324585813862, 'min_child_weight': 1.4239309238323308e-07, 'cat_smooth': 52}
r2:  0.9957284729138943
시간 : 27.041608333587646
6호선
Best parameters found:  {'learning_rate': 0.14183362867565227, 'n_estimators': 164, 'num_leaves': 14, 'max_depth': 3, 'min_child_samples': 5, 'subsample': 0.9944587790752809, 'colsample_bytree': 0.8796135321929832, 'reg_alpha': 0.14520332815850664, 'reg_lambda': 0.15680698915037036, 'min_split_gain': 4.1388625505187435e-06, 'min_child_weight': 0.031511128203972255, 'cat_smooth': 62}
r2:  0.9194240571878578
시간 : 21.860867738723755
7호선
Best parameters found:  {'learning_rate': 0.16002677822406342, 'n_estimators': 91, 'num_leaves': 16, 'max_depth': 4, 'min_child_samples': 5, 'subsample': 0.6752017008130472, 'colsample_bytree': 0.7654554630663708, 'reg_alpha': 0.8487160516386802, 'reg_lambda': 0.24146861293843727, 'min_split_gain': 0.0001922629775035098, 'min_child_weight': 0.3489574406352572, 'cat_smooth': 21}
r2:  0.9982686943454835
시간 : 29.08221459388733
8호선
Best parameters found:  {'learning_rate': 0.2528480988916025, 'n_estimators': 101, 'num_leaves': 12, 'max_depth': 7, 'min_child_samples': 5, 'subsample': 0.9002690784879681, 'colsample_bytree': 0.9424600012513639, 'reg_alpha': 0.537018410874369, 'reg_lambda': 21.68766162281525, 'min_split_gain': 0.004385019124269766, 'min_child_weight': 3.232175564353623e-06, 'cat_smooth': 14}
r2:  0.9869780121130545
시간 : 23.126200199127197
'''
