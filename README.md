今年はしっかりバージョン管理しよう

cd ./profit_naacl_new　下で実行

resume_folder_pathの例：/home/fukuda/output/Humanoid-v2-run226

テスト実行例：(Profit-naacl) fukuda@deepstation:~/M1_reserch/kc_profit_naacl_m1/profit_naacl_new_fortest$ python main_2.py --test_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-30_20_7_5_384.pkl" --train_pickle="../../datasets/pickles_0tumeru/S_test_data_sorted_in_jsons_6_2015-10-01~2015-10-30_20_7_5_384.pkl" --initial_account_balance=100000 --mode=test --resumes_folder_path="../../output_2/Humanoid-v2-run8"

Humanoid-v2-run8"をHumanoid-v2-run8/step_27200"でも実行できるよう変更

↑for_testと分かれてるの面倒　あと
INPUT_TEXT = "tweetonly" #tweetonly withSEC withtimefeatsSEC
USING_MODEL = "model_2" #
TRANSACTION_FEE_PERCENT = 0.000495 #0.000495
をconfigs_stock.py側で変更するのも面倒　なぜ引数にしていない？