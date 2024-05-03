import pandas as pd
from tqdm import tqdm
import requests
from datetime import datetime,timedelta


def update_data(loading_dir='/content/drive/MyDrive/Topic Mining Project/new_data/'): ## change to your path
    TOKEN = '67b0372029c40988b5df783c6f4c2cb49f9f82a2'
    HEADERS = {'Content-Type': 'application/json'}

    requestResponse = requests.get(f"https://api.tiingo.com/tiingo/news?&limit=1000&token={TOKEN}", headers=HEADERS)
    df=pd.DataFrame(requestResponse.json())
    if type(df.iloc[0]['tickers'])==str:
        df=df[len(df['tickers'])!='[]']
    elif type(df.iloc[0]['tickers'])==list:
        df=df[df['tickers'].apply(lambda x: len(x)>0)]
    else:
        print('tickers type is not str/list')
        raise ValueError

    df['publishedDate']=df['publishedDate'].apply(lambda x: x[:10])
    df.drop_duplicates(subset='id',inplace=True)


    df0=pd.read_csv(loading_dir+'news_data/news_with_ticker.csv',index_col=0)
    index_set=set(df0['id'])
    df=df[df['id'].apply(lambda x: x not in index_set)]
    pd.concat([df0,df]).to_csv(loading_dir+'news_data/news_with_ticker.csv')

def first_download(loading_dir='/content/drive/MyDrive/Topic Mining Project/new_data/'):
    start_date = datetime(2024, 1, 10)
    current_date = datetime.now()
    delta = current_date - start_date
    date_list=[]
    for i in range(delta.days + 1):
        date = start_date + timedelta(days=i)
        date_list.append(date.strftime("%Y-%m-%d"))

    TOKEN = '67b0372029c40988b5df783c6f4c2cb49f9f82a2'
    HEADERS = {'Content-Type': 'application/json'}
    data=pd.DataFrame()
    for date,next_date in tqdm(zip(date_list[:-1],date_list[1:]),total=len(date_list)-1):
        requestResponse = requests.get(f"https://api.tiingo.com/tiingo/news?startDate={date}&endDate={next_date}&limit=10000&token={TOKEN}", headers=HEADERS)
        df=pd.DataFrame(requestResponse.json())
        if len(df)==0:
            continue
        if type(df.iloc[0]['tickers'])==str:
            df=df[len(df['tickers'])!='[]']
        elif type(df.iloc[0]['tickers'])==list:
            df=df[df['tickers'].apply(lambda x: len(x)>0)]
        else:
            raise ValueError

        df['publishedDate']=df['publishedDate'].apply(lambda x: x[:10])
        df.drop_duplicates(subset='id',inplace=True)
        df=df[df['publishedDate']==date]
        data=pd.concat([data,df])

    data.to_csv(loading_dir+'news_data/news_with_ticker.csv')
    

if __name__=='__main__':
    update_data()