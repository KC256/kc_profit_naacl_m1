#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess
import os
from pathlib import Path
import time
import psutil


# In[30]:


#
def do_test(train_log_path):
    print(os.path.basename(train_log_path))
    with open(train_log_path, 'r') as file: #outputパスの取得
        lines = file.readlines()
        search_strings = ["output:", "INPUT_TEXT:", "USING_MODEL:"]
        key_values = []
        for i, search_string in enumerate(search_strings):
            for line in lines:
                if search_string in line:
                    temp = line.split(':')
                    temp = temp[1].replace(" ", "")
                    temp = temp.replace("\n", "")
                    key_values.append(temp)
                    break
            
        print(key_values)
        resumes_folder_path = os.path.join("/home/fukuda/", key_values[0])
        input_text = key_values[1]
        using_model = key_values[2]
        
        if input_text=="tweetonly":
            test_pickles_folder = "/home/ubuntu/datasets/pickles_0tumeru/for_test"
            test_pickles = os.listdir(test_pickles_folder)
            
        if input_text=="withtimefeatsSEC":
            test_pickles_folder = "/home/fukuda/datasets/pickles_withSECtimefeature_0tumeru/for_test"
            test_pickles = os.listdir(test_pickles_folder)
            
        if input_text=="withSEC":
            test_pickles_folder = "/home/ubuntu/datasets/pickles_oneSEC_0tumeru/for_test" #SECが1か２かでここのパスを/home/ubuntu/datasets/pickles_withSEC_0tumeruに変える必要
            test_pickles = os.listdir(test_pickles_folder)
            
        # elif input_text=="withSEC":
        #     test_pickles = "/home/ubuntu/datasets/pickles_0tumeru/for_test"
        
        initial_account_balances = [10000, 100000, 1000000, 10000000] #ここでテストしたいinitial_account_balance設定
        i=0
        for initial_account_balance in initial_account_balances:
            for test_pickle in test_pickles:
                try:
                    test_pickle = os.path.join(test_pickles_folder, test_pickle)
                    # print(test_pickle)
                    output_log_folder = "/home/fukuda/nohup_test_outputs_m2_212_2"
                    output_log_name = os.path.basename(train_log_path)[:-4] + ":" + str(initial_account_balance) + ":" + test_pickle[-36:-15]
                    output_log_path = os.path.join(output_log_folder, output_log_name)
                    
                    # counter = 1
                    # while os.path.exists(output_log_path + ".log"): #すでにファイルが存在している場合通し番号を追加
                    #     counter += 1
                    #     output_log_path = f"{output_log_path}:{counter}"
                    
                    j=0
                    while os.path.exists(output_log_path + ".log"): #すでにファイルが存在している場合スキップ　上と選択
                        print("already exist", output_log_name)
                        j=1
                        break
                        
                    output_log_path = output_log_path + ".log"
                    
                    if j==0:
                        command = f"setsid nohup python /home/fukuda/kc_profit_naacl/profit_naacl_new_fortest/main_2.py \
                                    --test_pickle={test_pickle} \
                                    --train_pickle={test_pickle} \
                                    --initial_account_balance={initial_account_balance} --mode=test \
                                    --resumes_folder_path={resumes_folder_path} \
                                    > {output_log_path} 2>&1 &"
                        
                        # print("not exist")
                        print(output_log_name)
                        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
                        if i%5==4:
                                print("sleep memory usage:", psutil.virtual_memory().percent)
                                time.sleep(50)
                        i+=1
                        while True:
                            if psutil.virtual_memory().percent > 50: #メモリ使用率が60%以上で
                                print("msleep memory usage:", psutil.virtual_memory().percent)
                                time.sleep(10)
                            else:
                                break
                    
                except Exception as e:
                    print("error:", e)
                
                
        
        


# In[ ]:


do_test("/home/fukuda/nohup_outputs_m2_212/with2SEC_tf_model4_2.log")




