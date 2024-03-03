from passenger_predict_torch_votingEnsemble import passenger_predict
from getoff_predict_torch_votingEnsemble import getoff_predict
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
'''
입력: 시작역 - 환승역 ... 환승역 - 도착역
1. 각 역간의 이동 문제로 쪼개기
2. 부분문제의 출발역에서 승차대기시간 구하기
2-1. 출발역 시점에서 지하철이 포화되어있는지 판단하기
2-2. 포화되었다면 얼마만큼 대기해야하는지 계산
3. 각 부분문제에서 노선 자체의 지연시간 가져오기
4. 각 시간들을 합하여 실질지연시간 구하기
'''
def is_max_at_station(departure_station:int,arrival_station:int,date:datetime=datetime.today())->bool:
    '''
    출발역과 도착역을 적으면 탑승시 열차가 가득차있는지 확인해줍니다
    '''
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
    
    ascending = True  # 역번호가 증가하는 방향이면 True, 감소하는 방향이면 False
    target_line = line_list[line_num]
    departure_station_idx = int(np.where(line_list[line_num] == departure_station)[0])
    arrival_station_idx = int(np.where(line_list[line_num] == arrival_station)[0])
    if departure_station_idx > arrival_station_idx: # 하행
        ascending = False
    # 상행인지 하행인지에 따라 더해야하는 역을 정함
    target_stations = (target_line[:departure_station_idx] if ascending else target_line[departure_station_idx:]) 
    
    date = str(date)[:14] + '00:00' # 시간만 필요하므로 분과 초는 날림
    full_time_list = list(pd.date_range("2023-01-01 00:00","2023-08-31 23:00",freq='h').astype(str))
    date_idx = full_time_list.index(date)
    
    total_passenger = 0
    for station in target_stations:
        passenger, _, __ = passenger_predict(station)
        print(passenger[date_idx])
        total_passenger += int(passenger[date_idx])
        total_passenger = total_passenger if total_passenger >= 0 else 0 # total_passenger가 음수면 0으로 대체
        print(f"{station} total_passenger {total_passenger}")

    trans_max_list = [2800,2800,2800,2800,2240,2240,2240,1680] # 호선별 최대 수송인원
    
    ''' 
    1이면 낮은쪽부터, -1이면 큰쪽부터 가까운 역까지 승객 변동을 더하기 
    그리고 호선 별 편성당 수송인원과 비교해서 만석인지 아닌지 구하기    
    '''
    # print("total passenger",total_passenger)
    isMax = False
    if total_passenger > trans_max_list[line_num]:
        isMax = True
    
    return isMax

def decode_interval_csv(line_num)->pd.DataFrame:    # 데이터가 바뀌니 다시 만들어야함
    '''
    배차시간 CSV파일 해독하는 함수
    입력값: 호선 번호
    반환값: DataFrame
    
    '''
    interval_csv = pd.read_csv('./data/상행선배차간격중위값_수정.csv')
    interval_csv = interval_csv.fillna(method='ffill')
    interval_csv = interval_csv.astype(str)
    for idx, data in enumerate(interval_csv['Hour']): # 8:00:00 형태를 08:00:00로 바꿈
        if len(data) == 7:
            data = '0' + data
            # print(interval_csv['Hour'][idx],data)
            interval_csv["Hour"][idx] = data
            
    interval_csv = interval_csv.set_index('Hour')
    interval_csv = interval_csv[f'{line_num}호선']
    # print(interval_csv)
    for idx, data in enumerate(interval_csv.values): # 시간을 초단위로 전환
        data = str(data).split(':')
        sec = 0
        for i, n in enumerate(data):
            sec += round(float(n)) * (60 ** (2-i))
        interval_csv.iloc[idx] = sec
        
    return interval_csv


def getin_delay_predict(departure_station:int,arrival_station:int,date=datetime.today())->int:
    '''
    param
        departure_station:int 출발역번호
        arrival_station:int   도착역번호
        date=datetime.today() 승차시간
    return
        int 지연시간(초)
    '''
    
    is_full_train = is_max_at_station(departure_station,arrival_station,date)
    if not is_full_train:
        return 0
    time = str(date).split()[1]
    
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
    
    interval = decode_interval_csv(line_num)
    interval_time = interval.loc[time]
    print(interval_time)
    
    full_time_list = list(pd.date_range("2023-01-01 00:00","2023-08-31 23:00",freq='h').astype(str))
    time_idx = full_time_list.index(date)
    
    passenger,_,__ = passenger_predict(departure_station)
    getoff_passenger,_,__ = getoff_predict(departure_station)
    passenger = passenger[time_idx]
    getoff_passenger = getoff_passenger[time_idx]
    getin_pessenger = getoff_passenger + passenger
    print(passenger,getin_pessenger,getoff_passenger,sep='------\n')
    
    lost_train_num = int(getin_pessenger / getoff_passenger) # 버림
    # lost_train_num = 2
    delay_time = interval_time * lost_train_num
    
    return delay_time

if __name__ == '__main__':
    # result = is_max_at_station(2827, 2828, "2023-04-04 08:00:00")
    # print(result)
    # print(decode_interval_csv(1))
    getin_delay = getin_delay_predict(2822, 2828, "2023-04-04 08:00:00")
    print("getin delay",getin_delay)
    pass