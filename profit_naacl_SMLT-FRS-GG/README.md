
粒度の考慮
20250911以降



cd ./profit_naacl_new　下で実行

resume_folder_pathの例：/home/fukuda/output/Humanoid-v2-run226

テスト実行例：(Profit-naacl) fukuda@deepstation:~/M1_reserch/kc_profit_naacl_m1/profit_naacl_new_fortest$ python main.py --test_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --train_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --initial_account_balance=100000 --mode=test --resumes_folder_path="../../output_2/Humanoid-v2-run8/step_27200"

--resumes_folder_path="../../output_2/Humanoid-v2-run8"でも実行できる

バックグラウンドテスト実行例：(Profit-naacl) fukuda@deepstation:~/M1_reserch/kc_profit_naacl_m1/profit_naacl_new_fortest$setsid nohup python main.py --test_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --train_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --initial_account_balance=100000 --mode=test --resumes_folder_path="../../output_2/Humanoid-v2-run8/step_27200" > temp_test_output.log 2>&1 & 

↑for_testと分かれてるの面倒　あと
INPUT_TEXT = "tweetonly" #tweetonly withSEC withtimefeatsSEC
USING_MODEL = "model_2" #
TRANSACTION_FEE_PERCENT = 0.000495 #0.000495
をconfigs_stock.py側で変更するのも面倒　引数にする

訓練実行例：(Profit-naacl) fukuda@deepstation:~/M1_reserch/kc_profit_naacl_m1/profit_naacl_new_fortest$ python main.py --test_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --train_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --initial_account_balance=100000 --mode=train --warmup=1 --train_iter=3 --seed=1

バックグランド訓練例：(Profit-naacl) fukuda@deepstation:~/M1_reserch/kc_profit_naacl_m1/profit_naacl_new_fortest$ setsid nohup python main.py --test_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --train_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-03_20_7_5_384.pkl" --initial_account_balance=100000 --mode=train --warmup=1 --train_iter=3 --seed=1 > temp_train_output.log 2>&1 &

./output_2はb4時に学習させたモデルのデータ(AWS),スクショからどれがどんなモデルか判断して