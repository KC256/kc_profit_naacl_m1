# shares normalization factor
# 100 shares per trade
HMAX_NORMALIZE = 10 #100
# initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 100000 #100000
# total number of stocks in our portfolio
# STOCK_DIM = 88
STOCK_DIM = 20 #修正 87 12
# N_DAYS = 7
N_DAYS = 7 #修正
PRICE_HISTORY_DAYS = 7  # ★★★ 新規追加: 過去7日間のデータを指定 ★★★
PRICE_FEATURES = 1      # ★★★ 新規追加: 終値のみなので1 ★★★
# MAX_TWEETS = 30
MAX_TWEETS = 5 #修正
MAX_SECS = 1 #1targetday1あたりのsecの数
MAX_LEN = MAX_TWEETS #追加 maxlenの値
TWEETS_EMB = 384 #768 4096 384
TIME_FEATS = STOCK_DIM * N_DAYS * MAX_TWEETS #ツイート間の時間差
SEC_TIME_FEATS = STOCK_DIM * MAX_SECS #SECの決算報告書をtarget_dayの日付差

INPUT_TEXT = "SMLT-FRS-GG" #tweetonly withSEC withtimefeatsSEC SMLT-FRS-G
USING_MODEL = "model_12" #model_2, model_4, model_3，model_5(SMLT-FRS株価使用)
TRANSACTION_FEE_PERCENT = 0.000495 #0.000495

SELECT_ACTION ="default" #ddpgのselect_actionにてどのようにactionを出力するか（ランダムアクション部）
#default random plus-minus-change

#     )
if(INPUT_TEXT=="SMLT-FRS-G"):
    FEAT_DIMS = (
        1
        + STOCK_DIM * 3
        + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB
        + STOCK_DIM * N_DAYS
        + 5 * STOCK_DIM
        + TIME_FEATS
        + STOCK_DIM * TWEETS_EMB * MAX_SECS #sec追加したので辻褄合わせ
        + STOCK_DIM * MAX_SECS
        + STOCK_DIM * PRICE_HISTORY_DAYS * PRICE_FEATURES
        + STOCK_DIM # last_price_2
        + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB #'embedding_2
        + STOCK_DIM * N_DAYS #'length_data_2'
        + TIME_FEATS # 'time_features_2'
    )

if(INPUT_TEXT=="SMLT-FRS-GG"):
    FEAT_DIMS = (
        1
        + STOCK_DIM * 3
        + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB
        + STOCK_DIM * N_DAYS
        + 5 * STOCK_DIM
        + TIME_FEATS
        + STOCK_DIM * TWEETS_EMB * MAX_SECS #sec追加したので辻褄合わせ
        + STOCK_DIM * MAX_SECS
        + STOCK_DIM * PRICE_HISTORY_DAYS * PRICE_FEATURES
        + STOCK_DIM # last_price_2
        + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB #'embedding_2
        + STOCK_DIM * N_DAYS #'length_data_2'
        + TIME_FEATS # 'time_features_2'
        + STOCK_DIM * PRICE_HISTORY_DAYS * PRICE_FEATURES # 'closing_price_5d_2'
    )


LAST_PRICE_IDX = 1
HOLDING_IDX = LAST_PRICE_IDX + STOCK_DIM
EMB_IDX = HOLDING_IDX + STOCK_DIM
LEN_IDX = EMB_IDX + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB
TARGET_IDX = LEN_IDX + STOCK_DIM * N_DAYS
PRICEDIFF_IDX = TARGET_IDX + STOCK_DIM
VOLDIFF_IDX = PRICEDIFF_IDX + STOCK_DIM
TEXTDIFF_IDX = VOLDIFF_IDX + STOCK_DIM
PRICE_TEXT_DIFF_IDX = TEXTDIFF_IDX + STOCK_DIM
ALLDIFF_IDX = PRICE_TEXT_DIFF_IDX + STOCK_DIM
TIME_IDX = ALLDIFF_IDX + STOCK_DIM
SECEMB_IDX = TIME_IDX + STOCK_DIM * MAX_TWEETS * N_DAYS
SECTIME_IDX = SECEMB_IDX + STOCK_DIM * TWEETS_EMB * MAX_SECS #withtimefeatsSEC
PRICE_HISTORY_IDX = SECTIME_IDX + STOCK_DIM * MAX_SECS
LAST_PRICE2_IDX = PRICE_HISTORY_IDX + STOCK_DIM * PRICE_HISTORY_DAYS * PRICE_FEATURES #SMLT-FRS-G
EMB2_IDX = LAST_PRICE2_IDX + STOCK_DIM
LEN2_IDX = EMB2_IDX + STOCK_DIM * N_DAYS * MAX_TWEETS * TWEETS_EMB
TIME2_IDX = LEN2_IDX + STOCK_DIM * N_DAYS
PRICE_HISTORY2_IDX = TIME2_IDX + STOCK_DIM * MAX_TWEETS * N_DAYS
# print("FEAT_DIMS:", FEAT_DIMS)

REWARD_SCALING = 1e-4

# print(TEXTDIFF_IDX)

#自作
IS_TRAINING = True #agent.is_trainig　トレーニング中かeval中か判断
import pandas as pd
global global_df
global_df = pd.DataFrame(columns=['day', 'step', 'value_loss'])