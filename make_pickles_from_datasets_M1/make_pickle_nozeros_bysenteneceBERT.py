#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer
import nltk
import os
import pandas as pd
from datetime import datetime, timedelta
import json
# from transformers import BertTokenizer, BertModel
import torch
import pickle
import numpy as np
import sys
import os
import torch
from torch.autograd import Variable
# import pandas as pd
from sentence_transformers import SentenceTransformer
# import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')


# In[ ]:


#pickleファイルの作成
#初期設定
# date_start = "2014-01-02" #作成するpickleファイルの開始日
# date_end = "2014-01-03" #作成するpickleファイルの終了日

# date_starts = ["2015-01-01", "2015-02-01"]
# date_ends = ["2015-03-31", "2015-04-30"]
# for date_start, date_end in zip(date_starts, date_ends):
#     print("Start Date:", date_start)
#     print("End Date:", date_end)

num_stocks = 20
lookback_length = 7
num_texts_per_day = 5 #ツイートが３０以上ある場合その箇所は省かれる
embedding_size = 384

folder_path = "/home/fukuda/profit-naacl/profit-naacl/test_data_sorted_in_jsons_6" #pickleファイル作成の対象となるフォルダ
print("folder_path ", folder_path)
parts = folder_path.split('/')
folder_name = parts[-1]

stock_price_folder = "/home/fukuda/stocknet-dataset/price/raw"

text_difficulty = torch.ones(num_stocks)
volatility = torch.ones(num_stocks)
price_text_difficulty = torch.ones(num_stocks)
price_text_vol_difficulty = torch.ones(num_stocks)
price_difficulty = torch.ones(num_stocks)
#ここまで初期設定


# In[2]:


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


# In[3]:


#token(1ツイート)に対し７６８ベクトルを返す
def get_average_embedding_output(tokens):
    result = " ".join(tokens)
    embeddings = model.encode(result)
    embeddings = embeddings.tolist()
    return embeddings


# In[4]:


#file_pathのjson_length, average_output, time_differenceを返す
def get_embedding_length_timefeatures(file_path, num_texts_per_day):

    # ll_json_length_array = np.zeros((len(stock_names), date_length))
    # average_output = np.empty(num_texts_per_day)
    # time_difference = np.zeros(num_texts_per_day)

    # average_output = [0]*num_texts_per_day #ツイートが一日３０件未満の場合その箇所の値は０
    embedding_size=384
    average_output = [[0 for _ in range(embedding_size)] for _ in range(num_texts_per_day)]
    # print(len(average_output[1]))
    time_difference = [[0]]*num_texts_per_day #ツイートが一日３０件未満の場合その箇所の値は０
    json_length = 0
    try:
        with open(file_path, 'r') as json_file:
            for line in json_file:
                if(json_length<num_texts_per_day): #ツイートが1日num_text_per_day件以上の場合num_text_per_day+1件目以降はパス
                    data_dict = json.loads(line) 
                    # print(data_dict["text"])
                    tokens = data_dict["text"]
                    average_output[json_length] = get_average_embedding_output(tokens)
                    # print(average_output)

                    # print(data_dict["created_at"])
                    now_datetime_obj = datetime.strptime(data_dict["created_at"], "%a %b %d %H:%M:%S %z %Y")
                    if(json_length>0):
                        time_difference_object = now_datetime_obj - past_datetime_obj
                        time_difference[json_length] = [int(time_difference_object.total_seconds())]
                        # print(f"時間差: {time_difference}")
                    else:#第一要素の場合時間差０とする
                        time_difference[json_length] = [0]
                    past_datetime_obj = now_datetime_obj
                    json_length += 1
    except FileNotFoundError:
        print(file_path, "does not exist")
        pass
    
    # average_output_tensor = to_tensor(average_output)
    return json_length, average_output, time_difference
    
# json_length, average_output, time_difference = get_embedding_length_timefeatures("/home/fukuda/profit-naacl/profit-naacl/test_data_sorted_2/AMZN/2014-01-02", 30)
# print(time_difference)


# In[5]:


#stockname date_targetからそれに対応する日付の企業のadj_closeを得る　ソースは/home/fukuda/AI4Finance-Foundation/stocknet-dataset/price/preprocessed
def get_adj_close(stock_name, date_target, date_target_object, stock_price_folder):
    stock_file = stock_price_folder + "/" + stock_name + ".csv"
    df = pd.read_csv(stock_file)
    
    target_data = pd.DataFrame()
    while(target_data.empty): #その日付が存在するまで
        # print(date_target)
        target_data = df[df["Date"] == date_target]
        date_target_object -= timedelta(days=1) #その日の終値が存在しない場合1日前の終値を代入する　一日前も存在しない場合は二日前。。
        date_target = date_target_object.strftime("%Y-%m-%d")
    adj_close = target_data["Adj Close"].iloc[0]

    return adj_close


# date_target = "2014-01-02"
# date_target_object= datetime.strptime(date_target, "%Y-%m-%d")
# get_adj_close("AAPL", date_target, date_target_object, "/home/fukuda/AI4Finance-Foundation/stocknet-dataset/price/raw")


# In[6]:


# #pickleファイルの作成
# date_start = "2015-01-08" #作成するpickleファイルの開始日
# date_end = "2015-01-30" #作成するpickleファイルの終了日

# num_stocks = 87
# lookback_length = 7
# num_texts_per_day = 30 #ツイートが３０以上ある場合その箇所は省かれる
# embedding_size = 768

# folder_path = "/home/fukuda/profit-naacl/profit-naacl/test_data_sorted_2" #pickleファイル作成の対象となるフォルダ
# folder_name = "test_data_sorted_3"
# stock_price_folder = "/home/fukuda/AI4Finance-Foundation/stocknet-dataset/price/raw"

# text_difficulty = torch.ones(num_stocks)
# volatility = torch.ones(num_stocks)
# price_text_difficulty = torch.ones(num_stocks)
# price_text_vol_difficulty = torch.ones(num_stocks)
# price_difficulty = torch.ones(num_stocks)

#----ここまで初期設定-------
def make_pickle(date_start, date_end):
    date_start_object= datetime.strptime(date_start, "%Y-%m-%d")
    date_end_object= datetime.strptime(date_end, "%Y-%m-%d")
    date_before_start_object = date_start_object - timedelta(days=lookback_length) #date_start からlookbacklength前の日
    date_before_start = date_before_start_object.strftime("%Y-%m-%d")
    date_target = date_before_start
    date_target_object = date_before_start_object
    stock_names = sorted(os.listdir(folder_path))
    date_length_object = date_end_object - date_before_start_object + timedelta(days=1)
    date_length = date_length_object.days
    # print(date_length, type(date_length))
    # sys.exit()

    # all_embedding_array = np.empty((len(stock_names), date_length), dtype=object)
    all_json_length_array = np.zeros((len(stock_names), date_length))
    # all_time_difference_array = np.empty((len(stock_names), date_length))

    # embedding_array = np.empty((len(stock_names), lookback_length), dtype=object)
    json_length_array = np.zeros((len(stock_names), lookback_length))
    # time_difference_array = np.empty((len(stock_names), lookback_length))
    # print(type(time_difference_array))

    all_time_difference_list = [[0] * date_length for _ in range(len(stock_names))]
    time_difference_list = [[None] * lookback_length for _ in range(len(stock_names))]
    # print(all_time_difference_list)
    # all_embedding_list = [[None] * date_length for _ in range(len(stock_names))]
    # embedding_list = [[None] * lookback_length for _ in range(len(stock_names))]

    all_embedding_list = [[[[0 for _ in range(embedding_size)] for _ in range(num_texts_per_day)] for _ in range(date_length)] for _ in range(num_stocks)]
    embedding_list =[[[[0 for _ in range(embedding_size)] for _ in range(num_texts_per_day)] for _ in range(lookback_length)] for _ in range(num_stocks)]



    #date_targetが最終日になるまで 各日の処理
    preprosessed_data = [None] * (date_length-lookback_length)
    j = 0 #日にちカウント
    k = [0]*len(stock_names) #all_embedding_list等カウント　日付詰めるため
    while(date_target_object != date_end_object + timedelta(days=1)):
        #dates, last_dateを出す
        date_last_object = date_target_object - timedelta(days=1)
        date_last = date_last_object.strftime("%Y-%m-%d")
        dates_object = []
        dates = []
        i = lookback_length
        while(i>0):
            dates_object.append(date_target_object- timedelta(days=i))
            temp = date_target_object- timedelta(days=i-1)
            dates.append(temp.strftime("%Y-%m-%d"))
            i -= 1

        adj_close_target = [None] * len(stock_names)
        adj_close_last = [None] * len(stock_names)
        i = 0
        #各企業に対する処理
        for stock_name in stock_names:
            # print(i)
            file_path =os.path.join(folder_path, stock_name, date_target) #01-02
            print(file_path)
            json_length, average_output, time_difference= get_embedding_length_timefeatures(file_path, num_texts_per_day) #これは当日のデータ
            adj_close_target[i]= get_adj_close(stock_name, date_target, date_target_object, stock_price_folder)
            adj_close_last[i] = get_adj_close(stock_name, date_last, date_last_object, stock_price_folder)
            # print(date_target, date_last, dates)
            # print(json_length, time_difference, average_output)
            # print(adj_close_last, adj_close_target)
            
            if os.path.exists(file_path) or j-lookback_length+1<=0: #[0]でない、または開始日前７日間のデータのみが入る
                all_embedding_list[i][k[i]] = average_output
                all_json_length_array[i][k[i]] = json_length
                all_time_difference_list[i][k[i]] = time_difference
                k[i] += 1
            
            if(j-lookback_length+1>0):
                embedding_list[i] = all_embedding_list[i][k[i]-lookback_length:k[i]]
                json_length_array[i] = all_json_length_array[i][k[i]-lookback_length:k[i]]
                time_difference_list[i] = all_time_difference_list[i][k[i]-lookback_length:k[i]]

            # print("len(embedding_list[i]) ", len(embedding_list[i]))
            i += 1
        
        if(j-lookback_length+1>0):
            # embedding_tensor = to_tensor(embedding_array)
            json_length_tensor = to_tensor(json_length_array)
            # print(type(json_length_tensor))
            # print(time_difference_list)
            # print(embedding_list)
            # print(type(torch.tensor(time_difference_list)), type(torch.tensor(embedding_list)))
            embedding_tensor = torch.tensor(embedding_list)
            time_difference_tensor = torch.tensor(time_difference_list)
            # time_difference_tensor = tf.convert_to_tensor(time_difference_array, dtype=tf.float32)
            adj_close_target_tensor = torch.tensor(adj_close_target)
            adj_close_last_tensor = torch.tensor(adj_close_target)
            # print("date target ", date_target)
            preprosessed_data[j-lookback_length] = {"dates":dates, "date_target":date_target, "date_last":date_last
                                        , "embedding":embedding_tensor, "length_data":json_length_tensor, "time_features":time_difference_tensor    
                                        , "adj_close_last":adj_close_last_tensor, "adj_close_target":adj_close_target_tensor, "text_difficulty":text_difficulty
                                        , "volatility":volatility, "price_text_difficulty":price_text_difficulty, "price_text_vol_difficulty":price_text_vol_difficulty
                                        , "price_difficulty":price_difficulty}

        date_target_object += timedelta(days=1)
        date_target = date_target_object.strftime("%Y-%m-%d")
        j += 1
        print(k)


    # In[7]:


    pickle_title = "S_" + folder_name  + "_" + date_start + "~" + date_end + "_" + str(num_stocks) + "_" + str(lookback_length)  + "_" + str(num_texts_per_day)  + "_" + str(embedding_size) + ".pkl"
    save_dir = "/home/fukuda/profit-naacl/profit-naacl/pickles_0tumeru"
    save_path = save_dir + "/" + pickle_title

    print(pickle_title)

    with open(save_path, "wb") as file:
        pickle.dump(preprosessed_data, file)


# In[11]:


# with open(save_path, "rb") as file:
#     loaded_data = pickle.load(file)
#     print(loaded_data[0])

# date_starts = ["2015-01-01", "2015-04-01", "2015-07-01", "2015-10-01"]
# date_ends = ["2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31"]
date_starts = ["2014-09-01"]
date_ends = ["2015-12-31"]
print("start")
for date_start, date_end in zip(date_starts, date_ends):
    print("Start Date:", date_start)
    print("End Date:", date_end)
    try:
        make_pickle(date_start, date_end)
    except Exception as e:
        print("エラーが発生しました:", e)