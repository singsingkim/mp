import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
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
    x1,x2,x3, y, train_size=0.99, random_state=100, stratify=y)

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
              early_stopping_rounds=10,
              verbose=False)
    
    y_pred = model.predict(x_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("RMSE:", rmse)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
e_t = time.time()
print("Best parameters found: ", study.best_params)
print("R2: ", study.best_value)
print("시간",e_t-s_t)
# 모델 저장이 잘 안되서 사용금지
# 모델 저장
model_path = 'c:/_data/_save/project/passenger+weather+bus_8호선.pkl'
params = study.best_params
model = XGBRegressor(**params, tree_method='gpu_hist')
model.fit(x_train, y_train)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

'''
1호선
Best parameters found:  {'learning_rate': 0.13349839953884737, 'n_estimators': 99, 'max_depth': 8, 'min_child_weight': 3.471164143831403e-06, 'subsample': 0.6661302167437514, 'colsample_bytree': 0.9856906281904222, 'gamma': 4.5485144879936555e-06, 'reg_alpha': 0.014276113125688179, 'reg_lambda': 10.121476098960851}
R2:  0.9999999996879733
시간 35.41777539253235
2호선
Best parameters found:  {'learning_rate': 0.23700707726857784, 'n_estimators': 84, 'max_depth': 3, 'min_child_weight': 2.1025806675510518e-05, 'subsample': 0.9978513616375926, 'colsample_bytree': 0.8740193270455657, 'gamma': 4.953590673563563e-07, 'reg_alpha': 0.2556901777099415, 'reg_lambda': 0.126023398338007}
R2:  0.9999999993383861
시간 33.462315797805786
3호선
Best parameters found:  {'learning_rate': 0.23062899771673678, 'n_estimators': 80, 'max_depth': 5, 'min_child_weight': 5.734126899926509e-06, 'subsample': 0.938818011060222, 'colsample_bytree': 0.947541730327353, 'gamma': 1.3636411785707547e-07, 'reg_alpha': 0.02065572134919358, 'reg_lambda': 0.020058199977467713}
R2:  0.9999999963107611
시간 30.213985919952393
4호선
Best parameters found:  {'learning_rate': 0.21118196368357803, 'n_estimators': 166, 'max_depth': 8, 'min_child_weight': 8.536212086356025e-06, 'subsample': 0.8372853675645959, 'colsample_bytree': 0.9594012891988645, 'gamma': 1.8455079878780107e-06, 'reg_alpha': 0.01777681522532268, 'reg_lambda': 1.221737676702471}
R2:  0.9999999999041183
시간 38.0778911113739
5호선
Best parameters found:  {'learning_rate': 0.2996671851691408, 'n_estimators': 188, 'max_depth': 7, 'min_child_weight': 3.56729794382612e-08, 'subsample': 0.765827475262291, 'colsample_bytree': 0.9588953646924626, 'gamma': 5.579693673619597e-06, 'reg_alpha': 0.04467575588717092, 'reg_lambda': 0.018087029021000043}
R2:  0.9999999994009818
시간 30.041101455688477
6호선
 Trial 99 finished with value: 0.0 and parameters: {'learning_rate': 0.015865462807948497, 'n_estimators': 80, 'max_depth': 4, 'min_child_weight': 0.008787322698730435, 'subsample': 0.7071932784521371, 'colsample_bytree': 0.7138038275878692, 'gamma': 47.74806785836774, 'reg_alpha': 21.704755881850353, 'reg_lambda': 0.35942746637631}. Best is trial 0 with value: 0.0.
Best parameters found:  {'learning_rate': 0.06984469597942058, 'n_estimators': 122, 'max_depth': 3, 'min_child_weight': 0.00014371216051494917, 'subsample': 0.5452487798087454, 'colsample_bytree': 0.7092572854505768, 'gamma': 0.19814393380840592, 'reg_alpha': 3.7362100200848594, 'reg_lambda': 0.08981603679640204}
R2:  0.0
시간 26.31062960624695
7호선
Best parameters found:  {'learning_rate': 0.2750234442782433, 'n_estimators': 162, 'max_depth': 4, 'min_child_weight': 0.00919359233412236, 'subsample': 0.9331773406590458, 'colsample_bytree': 0.9999982808922707, 'gamma': 2.727660008510403e-07, 'reg_alpha': 0.06947163364830904, 'reg_lambda': 0.11598693667315021}
R2:  0.99999999992449
시간 41.04697275161743
8호선
Best parameters found:  {'learning_rate': 0.20765802121945942, 'n_estimators': 193, 'max_depth': 6, 'min_child_weight': 2.0741193904507054e-06, 'subsample': 0.7823370007177193, 'colsample_bytree': 0.6714219298544106, 'gamma': 5.4374207875284934e-06, 'reg_alpha': 1.869092841479089, 'reg_lambda': 0.10847464956645747}
R2:  0.0
시간 24.844117164611816
'''
