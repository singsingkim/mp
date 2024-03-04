import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import make_passenger_csv, make_bus_csv, make_delay_csv

def passenger_bus_delay_corr():
    '''
    전철 인원, 버스 인원, 지연시간
    셋 데이터의 상관관계
    '''
    org = make_passenger_csv()
    cols = org.columns 

    passenger_data = pd.DataFrame()
    temp1 = pd.Series(sum(org[col] for col in cols[:10]))
    temp1 = round(temp1/10)
    temp2 = pd.Series(sum(org[col] for col in cols[10:60]))
    temp2 = round(temp1/50)
    temp3 = pd.Series(sum(org[col] for col in cols[60:94]))
    temp3 = round(temp1/34)
    temp4 = pd.Series(sum(org[col] for col in cols[94:120]))
    temp4 = round(temp1/26)
    temp5 = pd.Series(sum(org[col] for col in cols[120:176]))
    temp5 = round(temp1/56)
    temp6 = pd.Series(sum(org[col] for col in cols[176:215]))
    temp6 = round(temp1/39)
    temp7 = pd.Series(sum(org[col] for col in cols[215:264]))
    temp7 = round(temp1/49)
    temp8 = pd.Series(sum(org[col] for col in cols[264:282]))
    temp8 = round(temp1/18)

    print('========================쉐이프===========================')
    passenger_data['1호선'] = temp1
    passenger_data['2호선'] = temp2
    passenger_data['3호선'] = temp3
    passenger_data['4호선'] = temp4
    passenger_data['5호선'] = temp5
    passenger_data['6호선'] = temp6
    passenger_data['7호선'] = temp7
    passenger_data['8호선'] = temp8
    print('전철 쉐이프', passenger_data.shape)
    bus_data = make_bus_csv()
    print('버스 쉐이프', bus_data.shape)
    delay_data = make_delay_csv()
    print('지연 쉐이프', delay_data.shape)


    new_passenger_data = pd.DataFrame()
    for label in passenger_data:
        new_passenger_data[label] = passenger_data[label]
    for label in delay_data:
        new_passenger_data[label] = delay_data[label]
    for label in bus_data:
        new_passenger_data[label] = bus_data[label]
    
    print('전철 데이터\n', new_passenger_data.tail(24))

    new_passenger_matrix = new_passenger_data.corr()

    return new_passenger_matrix


# def plot_correlation_heatmap(new_passenger_matrix, bus_matrix):
def plot_correlation_heatmap(new_passenger_matrix): 
    
    new_passenger_matrix.to_csv('./data/new_passenger_matrix.csv')
    from matplotlib import font_manager, rc
    
    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    # 전철 상관관계 히트맵
    plt.figure(figsize=(12, 6))
    sns.heatmap(new_passenger_matrix, annot=True, cmap='coolwarm',
                fmt=".4f", linewidths=.5)
    # annot=True는 각 셀에 숫자 값을 표시하도록 하는 옵션입니다.
    # cmap='coolwarm'는 색상 맵을 지정합니다. 
    # fmt=".4f"는 셀에 표시되는 숫자의 형식을 소수점 네 번째 자리까지 표시하도록 지정합니다. 
    # linewidths=.5는 셀 사이의 선의 너비를 설정합니다. 
    # xticklabels=45는 x축 눈금 레이블이 매 45도마다 표시되도록 설정합니다
    # rotation=0은 x축 눈금 레이블의 텍스트가 수평으로 표시되도록 설정합니다.
    plt.title('전철 승객수 & 버스 승객수 & 전철지연시간 간의 상관관계')
    # plt.xlabel('Variables')
    # plt.ylabel('Variables')

    plt.tight_layout()
    plt.show()



# 데이터 준비 및 상관관계 계산
new_passenger_matrix = passenger_bus_delay_corr()
print(new_passenger_matrix.shape)

# 시각화
plot_correlation_heatmap(new_passenger_matrix)

