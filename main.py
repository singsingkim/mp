from calculate_delay import getin_delay_predict
from delay_predict_torch_votingEnsemble import delay_predict
from datetime import datetime
import pandas as pd
import numpy as np

def calculate_total_delay(departure_station:int,arrival_station:int,date=datetime.today())->int:
    
    # 어느 노선인지 확인
    if departure_station == arrival_station:
        raise Exception(f"departure and arrival is same num {departure_station}")
    line1 = np.arange(150,159+1)
    line2 = np.arange(201,250+1)
    line3 = np.arange(309,342+1)
    line4 = np.arange(409,434+1)
    line5 = np.arange(2511,2566+1)
    line6 = np.arange(2611,2648+1)
    line7 = np.arange(2711,2752+1)
    line8 = np.arange(2811,2828+1)
    line_list = [line1,line2,line3,line4,line5,line6,line7,line8]
    line_num = None
    for idx, line in enumerate(line_list):
        if departure_station in line:       # 출발역이 몇호선의 역인지 탐색
            if not(arrival_station in line):# 출발역과 도차역이 같은 호선이 아닌경우
                raise Exception(f"departure station{departure_station} and arrival station{arrival_station} is not same line")
            line_num = idx
            break
    else:   #출발역이 그 어떤 호선에도 존재하지 않는 역인 경우
        raise Exception(f"{departure_station}station is not exist")
    
    getin_delay = getin_delay_predict(departure_station,arrival_station,date)
    train_delay,_,__ = delay_predict(line_num)
    
    full_time_list = list(pd.date_range("2023-01-01 00:00","2023-08-31 23:00",freq='h').astype(str))
    time_idx = full_time_list.index(date)
    train_delay = train_delay[time_idx]
    
    final_delay = getin_delay + train_delay
    return final_delay

if __name__ == '__main__':
    departure_station, arrival_station = input("출발역과 도착역 번호를 적어주세요").split()
    date = input("원하는 날자와 시간을 적어주세요(2023-01-01부터 2023-08-31 23:00:00까지)")
    
    delay = calculate_total_delay(int(departure_station),int(arrival_station),date)
    print(f"총 지연시간은 {delay}초 입니다")