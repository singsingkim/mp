import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from preprocessing import load_bus, load_delay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import time
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_delay()

# 레이블 선택
bus = bus_csv
passenger = passenger_csv
weather = weather_csv
x1 = weather
x2 = delay_csv
# y = delay_csv['1호선지연(분)']
# y = delay_csv['2호선지연(분)']
# y = delay_csv['3호선지연(분)']
# y = delay_csv['4호선지연(분)']
# y = delay_csv['5호선지연(분)']
# y = delay_csv['6호선지연(분)']
# y = delay_csv['7호선지연(분)']
y = delay_csv['8호선지연(분)']

# print(pd.value_counts(y))

# 훈련 및 테스트 데이터 분할(원하는 상황으로 주석처리를 바꾸기)
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1,x2, y, train_size=0.99, random_state=100, stratify=y)

# 스케일링(모든 데이터 이용시)
scaler1 = StandardScaler()
scaler2 = MinMaxScaler()
scaler3 = RobustScaler()
# scaler = MaxAbsScaler()
x1_train_scaled = scaler1.fit_transform(x1_train)
x1_test_scaled = scaler1.transform(x1_test)

x2_train_scaled = scaler2.fit_transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)

# 스케일링(각각)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# 데이터 연결(모든 데이터 이용시)
x_train = np.concatenate((x1_train_scaled, x2_train_scaled), axis=1)
x_test = np.concatenate((x1_test_scaled, x2_test_scaled), axis=1)
print("x train shape", x_train.shape)
s_t = time.time()
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 67,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 100.0),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 100.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 100.0),
    }

    model = XGBRegressor(**params, tree_method='gpu_hist')
    model.fit(x_train, y_train,
              eval_set=[(x_test, y_test)],
              early_stopping_rounds=20,
              verbose=False)
    
    y_pred = model.predict(x_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE:", rmse)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
e_t = time.time()
print("Best parameters found: ", study.best_params)
print("R2: ", study.best_value)
print("시간",e_t-s_t)
# 모델 저장이 잘 안되서 사용금지
# 모델 저장
model_path = 'c:/_data/_save/project/mini_project_xgb_8호선.pkl'
params = study.best_params
model = XGBRegressor(**params, tree_method='gpu_hist')
model.fit(x_train, y_train)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

'''
1호선
Best parameters found:  {'learning_rate': 0.2218036245351803, 'n_estimators': 199, 'max_depth': 3, 'min_child_weight': 0.07709868781803283, 'subsample': 0.80309973945344, 'colsample_bytree': 0.9254025887963853, 'gamma': 6.628562492458777e-08, 'reg_alpha': 0.012998871754325427, 'reg_lambda': 0.10637051171111844}
R2:  0.9999998346366332
시간 31.593716144561768
2호선
Best parameters found:  {'learning_rate': 0.09016317170455039, 'n_estimators': 96, 'max_depth': 7, 'min_child_weight': 1.388365511671734, 'subsample': 0.6865717610042035, 'colsample_bytree': 0.9232135600837511, 'gamma': 0.0002540374277412407, 'reg_alpha': 0.13309380505107918, 'reg_lambda': 2.5793881973102013}
R2:  0.9999959235366351
시간 34.56555247306824
3호선
Best parameters found:  {'learning_rate': 0.1521220208110879, 'n_estimators': 167, 'max_depth': 3, 'min_child_weight': 4.2380242373519095e-07, 'subsample': 0.8444448351622212, 'colsample_bytree': 0.9667859379704544, 'gamma': 6.164430971076744e-07, 'reg_alpha': 0.0341565978919673, 'reg_lambda': 0.06826738806511448}
R2:  0.9999993272567578
시간 35.58358669281006
4호선
Best parameters found:  {'learning_rate': 0.2928930726197313, 'n_estimators': 56, 'max_depth': 4, 'min_child_weight': 1.8231358381881846e-07, 'subsample': 0.5793801911451185, 'colsample_bytree': 0.920043124958588, 'gamma': 3.755140180543055e-08, 'reg_alpha': 0.5734508972767961, 'reg_lambda': 0.06180573917831755}
R2:  0.9999999906161277
시간 23.021361827850342
5호선
Best parameters found:  {'learning_rate': 0.2970112648256843, 'n_estimators': 80, 'max_depth': 8, 'min_child_weight': 1.2539232696575079e-08, 'subsample': 0.50789458673495, 'colsample_bytree': 0.9473390574287422, 'gamma': 0.08696398230349235, 'reg_alpha': 0.011151412242781942, 'reg_lambda': 0.7762658151723353}
R2:  0.9999950793043366
시간 28.278682947158813
6호선
Trial 998 finished with value: 0.0 and parameters: {'learning_rate': 0.15758803501953397, 'n_estimators': 120, 'max_depth': 3, 'min_child_weight': 0.08022851302529604, 'subsample': 0.7469902109192953, 'colsample_bytree': 0.6160554114500078, 'gamma': 4.626179071331661e-06, 'reg_alpha': 0.5812933344208567, 'reg_lambda': 0.07527207384321995}. Best is trial 0 with value: 0.0.
RMSE: 0.006703829764923246
R2 Score: 0.0
7호선
Best parameters found:  {'learning_rate': 0.0483506866311774, 'n_estimators': 191, 'max_depth': 4, 'min_child_weight': 0.025433650473110674, 'subsample': 0.9800654867064976, 'colsample_bytree': 0.9323307630802589, 'gamma': 9.685770129589847e-05, 'reg_alpha': 1.7413155864823167, 'reg_lambda': 0.02616320673519039}
R2:  0.9999918494890756
RMSE: 0.009353158449144623
시간 36.50435709953308
8호선
Trial 998 finished with value: 0.0 and parameters: {'learning_rate': 0.056015044464279753, 'n_estimators': 77, 'max_depth': 8, 'min_child_weight': 0.5213441234702724, 'subsample': 
0.7431845077483091, 'colsample_bytree': 0.9227415521265859, 'gamma': 6.872043705762653e-08, 'reg_alpha': 0.37698627820948183, 'reg_lambda': 9.99460669905252}. Best is trial 0 with value: 0.0.
RMSE: 0.0009575860736170763
R2 Score: 0.0
'''
    

