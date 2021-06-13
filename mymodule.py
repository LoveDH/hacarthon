import time, json
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 고객별 차량 데이터 갯수를 세고, 1대 보유 고객과 2대 이상 보유 고객을 구분합니다.
# input: 차량 dataframe
# output: 1대보유고객 dataframe, 2대이상보유고객 dataframe, 전체 고객 수, 1대보유고객수
def count_car(df):
    
    len_cars = df['CUS_ID'].value_counts()
    single_car, multiple_car = df[df['CUS_ID'].map(len_cars==1)], df[df['CUS_ID'].map(len_cars>1)]
    
    single_car['LABEL'] = 0    # 1대만 보유한 고객들은 모두 신규구매자
    cus_count, count_add_single = len(set(df['CUS_ID'])), len(single_car)    # cus_count: 전체 고객 수, count: 현재까지의 고객 수
    
    print(count_add_single,'명의 고객은 차가 1대 있습니다.')
    
    return single_car, multiple_car, cus_count, count_add_single
    

# 고객별 차량출고일자와 보유종료일자를 dictionary 형태로 저장합니다.
# input: 2대이상보유고객 dataframe
# output: 고객id, 차량출고일자, 보유종료일자, 고객별 dictionary 데이터
def df_to_dict(multiple_car):
    cus_id, starts, finishes = list(multiple_car['CUS_ID']), list(multiple_car['WHOT_DT']), list(multiple_car['CAR_HLDG_FNH_DT'])
    
    car_dict = dict()
    for cid, st, fi in zip(cus_id, starts, finishes):
        car_dict[cid] = car_dict.get(cid, []) + [(st,fi)]
    cus_id = sorted(list(set(cus_id)))
    
    return cus_id, starts, finishes, car_dict


# 차량데이터에서 신규구매, 대차, 추가구매를 분리합니다. (0: 신규구매, 1: 대차, 2: 추가구매)
# input: 차량 dataframe, 진행도 로그 간격
# output: 레이블이 추가된 차량 dataframe
def decha_or_chugu(df, verbose=5000):
    start = time.time()
    
    single_car, multiple_car, cus_count, count = count_car(df)
    cus_id, starts, finishes, car_dict = df_to_dict(multiple_car)
    
    labels = []
    
    for cid in cus_id:
        label = [0]
        
        data = car_dict[cid]
        start_dates = [(num,'S',info[0]) for num, info in enumerate(data)]
        finish_dates = [(num,'F', info[1]) for num, info in enumerate(data)]

        data_merged = sorted(start_dates + finish_dates, key=lambda x:x[2])
        has_car = [0]
        
        for sequence in data_merged[1:]:
            has_fin = list(filter(lambda x: x[0] in has_car, finish_dates))
            
            if sequence[1]=='F':
                has_car.remove(sequence[0])
            elif sequence[1]=='S':
                has_car.append(sequence[0])
                has_fin_days_diff = [(sequence[2]-has_fin_date[2]).days for has_fin_date in has_fin if (sequence[2]-has_fin_date[2]).days<-51]
                all_fin_days_diff = [abs((sequence[2]-all_fin_date[2]).days) for all_fin_date in finish_dates if abs((sequence[2]-all_fin_date[2]).days)<=51]
                
                if all_fin_days_diff: # 구매 전후 51일간 보유종료이력이 있다면 대차
                    label.append(1)
                elif has_fin_days_diff: # 보유종료일자의 51일보다 먼저 차량을 구매했을 경우 추가구매
                    label.append(2)
                else:                   # 위에 해당사항이 없으면 신규구매
                    label.append(0)
        labels += label

        count+=1
        if count%verbose==0:
            print('진행도: ',count,'/',cus_count, '({}%)'.format(round(count/cus_count*100,1)))
    
    print('진행도: ',count,'/',cus_count, '({}%)'.format(round(count/cus_count*100,1)))
    multiple_car['LABEL'] = labels
    df = pd.concat([single_car, multiple_car]).sort_values(['CUS_ID','WHOT_DT'])
    print('\n로직 수행 시간:', round(time.time()-start,2), '초')
    
    return df


# 주어진 dataframe의 특정 feature에 대해 pie chart를 그립니다.
# input: dataframe, 그리고싶은 column명, 행 수, 열 수
# output: pie chart
def draw_pie(data, feature, row, col):
    classes = data[feature].unique()
    ccount = len(classes)
    fig = make_subplots(rows=row, cols=col, specs=[[{'type':'domain'}]*col]*row, subplot_titles=tuple(classes))

    for i, cls in enumerate(classes):
        df = data[data[feature]==cls]['LABEL'].value_counts()
        labels, values = df.index, df.values
        fig.add_trace(go.Pie(labels=labels, values=values, name=cls), i//5+1, i%5+1)
        
    fig.update_layout(template='seaborn')
    fig.update_traces(hoverinfo="label+percent+name")
    fig.show()


# 주어진 dataframe의 특정 feature에 대해 histogram를 그립니다.
# input: dataframe, 그리고싶은 column명
# output: histogram
def draw_hist(data, feature):
    df1, df2 = data[data['LABEL']=='대차'], data[data['LABEL']=='추가구매']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df1[feature], name='대차'))
    fig.add_trace(go.Histogram(x=df2[feature], name='추가구매'))

    fig.update_layout(barmode='overlay', template='seaborn')
    fig.update_traces(opacity=0.75)
    fig.show()


# 대차 비율을 도/광역시별로 지도에 표시한다.
# input: dataframe
# output: choropleth mapbox
def draw_map(data):
    state_geo = 'TL_SCCO_CTPRVN.zip.geojson'
    state_geo1 = json.load(open(state_geo, encoding='utf-8'))

    ratio_dict = dict()
    for do in data['CUS_ADM_TRY_NM'].unique():
        df = data[data['CUS_ADM_TRY_NM']==do]
        ratio = df['LABEL'].value_counts()['대차']/len(df)
        ratio_dict[do] = [round(ratio,4)]

    df= pd.DataFrame.from_dict(ratio_dict).T.reset_index().rename(columns={'index':'geo',0:'ratio'})
    fig = px.choropleth_mapbox(df,
                            geojson=state_geo1,
                            locations='geo',
                            color='ratio',
                            color_continuous_scale='RdBu',
                            range_color=(df['ratio'].min(),df['ratio'].max()),
                            mapbox_style='carto-positron',
                            featureidkey='properties.CTP_KOR_NM',
                            zoom=5.5,
                            center={'lat':35.565,'lon':127.486},
                            opacity=0.75)

    fig.update_layout(margin={'r':0,'t':0,'l':0,'b':0})
    fig.show()


# 재구매 유형을 시계열로 그린다.
# input: dataframe
# output: time series
def draw_car_ts(df):
    fig = px.line(df.groupby(['LABEL','YM']).count().reset_index(),
        x='YM',
        y='CAR_ID',
        color='LABEL')
    fig.update_layout(template='seaborn')
    fig.show()

