import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from keras.layers import concatenate
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
import time
# 데이터 로드
s_t = time.time()
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

def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        'iterations': trial.suggest_int('iterations', 150, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 1e-7, 10.0),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 10.0),
        'border_count': trial.suggest_int('border_count', 170, 350),
        'verbose': False,
        }

    # CatBoost 모델 생성(회귀)
    model = CatBoostRegressor(**params,devices='gpu')
    
    # 모델 학습
    model.fit(x_train, y_train,eval_set=[(x_test, y_test)],)
    
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)
    
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)
    
    return r2
#optuna를 사용해서 최적의 파라미터 찾기
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
#n_trials : 최적화를 위해 시도할 하이퍼파라미터 조합의 수
e_t = time.time()
print("Best parameters found: ", study.best_params)
print("r2: ", study.best_value)
print("시간",e_t-s_t)
# 최적의 모델 생성
best_params = study.best_params
best_model = CatBoostRegressor(**best_params,devices = 'gpu')

# 최적의 모델 학습
best_model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=False)

# 모델 저장
best_model.save_model('c:/_data/_save/project/passenger+weather+bus_catboost_8호선.pkl')

'''
1호선
Best parameters found:  {'iterations': 258, 'learning_rate': 0.10372129288289585, 'depth': 4, 'l2_leaf_reg': 7.502013869614672, 'bagging_temperature': 0.0006871304265196931, 'random_strength': 0.002522427032110104, 'border_count': 348}
r2:  0.9999999903254874
2호선
_temperature': 2.683625069463102, 'random_strength': 0.00026853744791303355, 'border_count': 238}. Best is trial 6 with value: 0.9979420948739732.
Best parameters found:  {'iterations': 263, 'learning_rate': 0.2578523597590685, 'depth': 8, 'l2_leaf_reg': 0.004066338025694324, 'bagging_temperature': 8.293666675333949e-07, 'random_strength': 0.002532746075400097, 'border_count': 330}
r2:  0.9979420948739732
시간 163.87230896949768
3호선
Best parameters found:  {'iterations': 250, 'learning_rate': 0.12961357134175788, 'depth': 5, 'l2_leaf_reg': 7.498471187130084e-05, 'bagging_temperature': 0.036817510877619426, 'random_strength': 1.026314729018688e-08, 'border_count': 323}
r2:  0.9999990790408725
시간 141.22464513778687
4호선
Best parameters found:  {'iterations': 268, 'learning_rate': 0.08241037174655148, 'depth': 4, 'l2_leaf_reg': 9.24763912756397, 'bagging_temperature': 0.025318089125076784, 'random_strength': 6.516728376449145e-07, 'border_count': 230}
r2:  0.9999942395144076
시간 125.40114045143127
5호선
Best parameters found:  {'iterations': 265, 'learning_rate': 0.11660384934788962, 'depth': 5, 'l2_leaf_reg': 0.0006352051946339236, 'bagging_temperature': 0.07296364453835182, 'random_strength': 0.0003146735937917479, 'border_count': 300}
r2:  1.0
시간 102.09353804588318
6호선
Best parameters found:  {'iterations': 227, 'learning_rate': 0.033424663973947666, 'depth': 6, 'l2_leaf_reg': 7.250925944260273e-07, 'bagging_temperature': 0.0003616111447379027, 'random_strength': 0.0011598027963551653, 'border_count': 217}
r2:  0.9611220387564399
시간 55.282793283462524
7호선
Best parameters found:  {'iterations': 200, 'learning_rate': 0.027167007114799187, 'depth': 4, 'l2_leaf_reg': 0.001772385358702794, 'bagging_temperature': 1.1477546571694337e-06, 'random_strength': 0.009738700964432667, 'border_count': 292}
r2:  0.9999843003199334
시간 97.15134143829346
8호선
Best parameters found:  {'iterations': 192, 'learning_rate': 0.10970561946454092, 'depth': 5, 'l2_leaf_reg': 1.6535967449696968e-06, 'bagging_temperature': 0.0040198556888896755, 'random_strength': 0.09179084511226566, 'border_count': 203}
r2:  0.9998742150879976
시간 139.48180150985718
'''
