from configs_stock import *
from empyrical import sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio
import pyfolio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import os

matplotlib.use("Agg")

class StockEnvTrade(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        all_data, #ここにpickleのデータが入る
        day, #ここに本来入るべきデータが入っていない(0)　もしくわself.all_data[0]となるようなデータ構造にする
        args,
        turbulence_threshold=140,
        initial=True,
        previous_state=[],
        model_name="", #保存時の名前指定
        iteration="", #保存時の名前指定
    ):
        """
        all_data: list containing the dataset.
        day: the date on which the agent will start trading.
        last_day: last_day in the dataset.
        """

        self.args = args
        
        #自作
        self.daybeforecash = INITIAL_ACCOUNT_BALANCE #一日前のcash
        self.daybefore_end_total_asset = np.zeros(STOCK_DIM) #一日前のend_total_asset
        self.buy_num = 0 #その日実際にかった企業数
        self.sell_num = 0 #その日実際に売った企業数
        self.hold_num = 0 #売買に含まれていたものの実際には取引しなかった企業数
        
        
        # print("self.daybeforecash", self.daybeforecash)
        # day = 0 #これダメ　
        self.day = day
        self.all_data = all_data

        # print(self.args)
        # print(self.day)
        # print(self.all_data)
        # print(type(self.all_data))
        # print(initial)
        
        self.data = self.all_data[self.day]
        # self.data = self.all_data #修正　作成者の想定しているdataが私が用意したall_dataに対応　今後複数のtarget_dayを持つようなデータセットの用意が必要

        self.initial = initial
        self.previous_state = previous_state

        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEAT_DIMS,)
        )
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # initalize state
        # print(type(self.data["adj_close_last"]), type(self.data["embedding"]), type(self.data["text_difficulty"]))
        # print(self.data["adj_close_last"])
        last_price = self.data["adj_close_last"].view(-1).tolist() #numpy出なくtensor型に
        target_price = self.data["adj_close_target"].view(-1).tolist()
        len_data = self.data["length_data"].view(-1).tolist()
        emb_data = self.data["embedding"].view(-1).tolist()
        text_diff = self.data["text_difficulty"].view(-1).tolist() #元のpickleファイルにはないキー
        vol_diff = self.data["volatility"].view(-1).tolist() #元のpickleファイルにはないキー
        price_text_diff = self.data["price_text_difficulty"].view(-1).tolist() #元のpickleファイルにはないキー
        price_diff = self.data["price_difficulty"].view(-1).tolist() #元のpickleファイルにはないキー
        all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist() #元のpickleファイルにはないキー
        time_feats = self.data["time_features"].view(-1).tolist() 
        self.state = ( #self.stateには元のpickleの各要素（つまりtarget_dayが一日分しか入っていない
            [INITIAL_ACCOUNT_BALANCE]  # balance
            + last_price  # stock prices initial
            + [0] * STOCK_DIM  # stocks on hold　ここの調整が必要か
            + emb_data  # tweet features
            + len_data  # tweet len
            + target_price  # target price
            + price_diff
            + vol_diff
            + text_diff
            + price_text_diff
            + all_diff
            + time_feats
        )
        # print(len(self.state), len(emb_data))
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # memorize all the total balance change
        # value of assets cash + shares
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        # biased CL rewards, during test all diff values are 1 so no change will happen in rewards
        self.rewards_memory = []
        # self.reset()
        self._seed()
        self.model_name = model_name
        self.iteration = iteration

    def _sell_stock(self, index, action):
        # perform sell action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            # check whether you have some stocks or not
            if self.state[index + STOCK_DIM + 1] > 0:
                # update balance by adding the money you get ater selling that particular stock
                self.state[0] += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                # update the number of stocks
                self.state[index + STOCK_DIM + 1] -= min(
                    abs(action), self.state[index + STOCK_DIM + 1]
                )
                self.cost += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * TRANSACTION_FEE_PERCENT
                )
                self.trades += 1
                self.sell_num += 1
            else:
                self.hold_num += 1
                pass
        else:
            # if turbulence goes over threshold, just clear out all positions
            if self.state[index + STOCK_DIM + 1] > 0:
                # update balance
                self.state[0] += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * TRANSACTION_FEE_PERCENT
                )
                self.trades += 1
                self.sell_num += 1
            else:
                self.hold_num += 1
                pass

    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        if self.turbulence < self.turbulence_threshold:
            # print(self.state[0], self.state[index + 1], index, self.state[1], self.state[index])
            # print(self.state)
            available_amount = self.state[0] // self.state[index + 1] #self.state[]が何を指すのか　STOCK_DIM = 2に変更
            # print('available_amount:{}'.format(available_amount))
            # update balance
            self.state[0] -= (
                self.state[index + 1]
                * min(available_amount, action)
                * (1 + TRANSACTION_FEE_PERCENT)
            )

            self.state[index + STOCK_DIM + 1] += min(available_amount, action)

            self.cost += (
                self.state[index + 1]
                * min(available_amount, action)
                * TRANSACTION_FEE_PERCENT
            )
            self.trades += 1
            self.buy_num += 1
        else:
            # if turbulence goes over threshold, just stop buying
            self.hold_num += 1
            pass

    def step(self, actions):
        # print(self.day, len(self.all_data))
        self.terminal = self.day >= len(self.all_data) - 1
        # print(actions)

        if self.terminal:
            print("Reached the end.")
            if not os.path.exists("results"):
                os.makedirs("results")
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                "results/account_value_trade_{}_{}.png".format(
                    self.model_name, self.iteration
                )
            )
            plt.close()
            print("self.asset_memory:", self.asset_memory)
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(
                "results/account_value_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )
            end_total_asset = self.state[0] + sum(
                np.array(self.state[HOLDING_IDX:EMB_IDX])
                * np.array(self.state[TARGET_IDX:PRICEDIFF_IDX])
            )
            print("previous_total_asset:{}".format(self.asset_memory[0]))

            print("end_total_asset:{}".format(end_total_asset))
            print("total_reward:{}".format(end_total_asset - self.asset_memory[0]))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ["account_value"]
            
            df_total_value["daily_return"] = df_total_value.pct_change(1) #pct_changeの使用で一行目がNan 増加率を出してる
            # df_total_value["daily_return"][0] = 0 #NaNを消しても解決しない　おそらく全ての値が０で標準偏差が０なのが問題
            # print("df_total_value:", df_total_value)
            
            #df_total_value["daily_return"]が全て同じ値でなければsharp_ratioは計算できる つまり何もしない状態が続いてたからNaNになる
            # df_total_value["daily_return"][3] = 0.1
            # df_total_value["daily_return"][2] = 0.2
            # df_total_value["daily_return"][4] = -0.1
            
            # print("df_total_value[daily_return]:", df_total_value["daily_return"])

            cum_returns = (
                (end_total_asset - self.asset_memory[0]) / (self.asset_memory[0])
            ) * 100
            
            # 一回全部消してみる
            sharpe = sharpe_ratio(df_total_value["daily_return"]) #ここの計算がうまくいってない
            sortino = sortino_ratio(df_total_value["daily_return"])
            calmar = calmar_ratio(df_total_value["daily_return"])
            mdd = max_drawdown(df_total_value["daily_return"]) * 100

            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv(
                "results/account_rewards_trade_{}_{}.csv".format(
                    self.model_name, self.iteration
                )
            )
            
            # print("sharpe(env.step):", sharpe)

            return (
                self.state,
                self.reward,
                self.terminal,
                {
                    #一回全部消してみる
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "calmar": calmar,
                    "mdd": mdd,
                    
                    # "sharpe": 1,
                    # "sortino": 1,
                    # "calmar": 1,
                    # "mdd": 1,
                    
                    "cum_returns": cum_returns,
                },
            )

        else:
            # print("actions ", actions)
            actions = actions * HMAX_NORMALIZE
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

            argsort_actions = np.argsort(actions)

            # print("actions ", actions)
            self.buy_num = 0 #その日実際にかった企業数
            self.sell_num = 0 #その日実際に売った企業数
            self.hold_num = 0 #売買に含まれていたものの実際には取引しなかった企業数
            
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
            
            # print("sell_index: ", sell_index)
            # print("buy_index: ", sell_index)

            begin_total_asset = np.array(self.state[HOLDING_IDX:EMB_IDX]) * np.array(
                self.state[LAST_PRICE_IDX:HOLDING_IDX] #ここが前日の値にならんとあかんのに現状今日の株価になってる
            )
            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])
                
            print("sell, buy, argsort: ", len(sell_index), len(buy_index), len(argsort_actions))
            print("self.sell_num, self.buy_num, self.hold_num:", self.sell_num, self.buy_num, self.hold_num)

            # print(np.array(self.state[HOLDING_IDX:EMB_IDX]), np.array(self.state[TARGET_IDX:PRICEDIFF_IDX]))
            # print(len(self.state))
            end_total_asset = np.array(self.state[HOLDING_IDX:EMB_IDX]) * np.array(
                self.state[TARGET_IDX:PRICEDIFF_IDX]
            )
            # print("sum(end_total_asset):", sum(end_total_asset))

            self.asset_memory.append(self.state[0] + sum(end_total_asset))

            if self.args.diff == "price":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[PRICEDIFF_IDX:VOLDIFF_IDX]) #利益を修正（重み的な）　次元は銘柄数
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "vol":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[VOLDIFF_IDX:TEXTDIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "text":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[TEXTDIFF_IDX:PRICE_TEXT_DIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "price_text":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[PRICE_TEXT_DIFF_IDX:ALLDIFF_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            elif self.args.diff == "pvt":
                self.reward = sum(
                    (end_total_asset - begin_total_asset)
                    * np.array(self.state[ALLDIFF_IDX:TIME_IDX])
                )
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            else:
                # self.reward = sum(end_total_asset - begin_total_asset) + self.state[0] - self.daybeforecash #+ self.state[0] - self.daybeforecashを追加
                self.reward = sum(end_total_asset - self.daybefore_end_total_asset) + self.state[0] - self.daybeforecash
                self.rewards_memory.append(self.reward)
                self.reward = self.reward * REWARD_SCALING
            # print("end_total_asset begin_total_asset", end_total_asset, begin_total_asset)
            # print("self.daybefore_end_total_asset", self.daybefore_end_total_asset)
            # print("sum end_total_asset sum begin_total_asset", sum(end_total_asset), sum(begin_total_asset))
            # print("self.state[0] ", self.state[0])
            # print("self.daybeforecash", self.daybeforecash)
            self.daybeforecash = self.state[0]
            self.daybefore_end_total_asset = end_total_asset
            # print("self.reward", self.reward/REWARD_SCALING)
                
            # print("2sum(end_total_asset):", sum(end_total_asset))

            self.day += 1
            # print("self.day ", self.day)
            self.data = self.all_data[self.day] #ここでtarget_dayを次の日にしている
            # self.data = self.all_data #修正

            last_price = self.data["adj_close_last"].view(-1).tolist()
            target_price = self.data["adj_close_target"].view(-1).tolist()
            len_data = self.data["length_data"].view(-1).tolist()
            emb_data = self.data["embedding"].view(-1).tolist()
            text_diff = self.data["text_difficulty"].view(-1).tolist()
            vol_diff = self.data["volatility"].view(-1).tolist()
            price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
            price_diff = self.data["price_difficulty"].view(-1).tolist()
            all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
            time_feats = self.data["time_features"].view(-1).tolist()
            self.state = (
                [self.state[0]]  # balance
                + last_price  # stock prices initial
                + list(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
                + emb_data  # tweet features
                + len_data  # tweet len
                + target_price  # target price
                + price_diff
                + vol_diff
                + text_diff
                + price_text_diff
                + all_diff
                + time_feats
            )
            
            # print("self.reward", self.reward)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        if self.initial:
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.all_data[self.day]
            # self.data = self.all_data #修正
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=self.iteration
            self.rewards_memory = []
            # initiate state
            last_price = self.data["adj_close_last"].view(-1).tolist()
            target_price = self.data["adj_close_target"].view(-1).tolist()
            len_data = self.data["length_data"].view(-1).tolist()
            emb_data = self.data["embedding"].view(-1).tolist()
            text_diff = self.data["text_difficulty"].view(-1).tolist()
            vol_diff = self.data["volatility"].view(-1).tolist()
            price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
            price_diff = self.data["price_difficulty"].view(-1).tolist()
            all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
            time_feats = self.data["time_features"].view(-1).tolist()
            self.state = (
                [INITIAL_ACCOUNT_BALANCE]  # balance
                + last_price  # stock prices initial
                + [0] * STOCK_DIM  # stocks on hold
                + emb_data  # tweet features
                + len_data  # tweet len
                + target_price  # target price
                + price_diff
                + vol_diff
                + text_diff
                + price_text_diff
                + all_diff
                + time_feats
            )
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.previous_state[1 : (STOCK_DIM + 1)])
                * np.array(self.previous_state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )
            self.asset_memory = [previous_total_asset]
            # self.asset_memory = [self.previous_state[0]]
            self.day = 0
            self.data = self.all_data[self.day]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            # self.iteration=iteration
            self.rewards_memory = []
            last_price = self.data["adj_close_last"].view(-1).tolist()
            target_price = self.data["adj_close_target"].view(-1).tolist()
            len_data = self.data["length_data"].view(-1).tolist()
            emb_data = self.data["embedding"].view(-1).tolist()
            text_diff = self.data["text_difficulty"].view(-1).tolist()
            vol_diff = self.data["volatility"].view(-1).tolist()
            price_text_diff = self.data["price_text_difficulty"].view(-1).tolist()
            price_diff = self.data["price_difficulty"].view(-1).tolist()
            all_diff = self.data["price_text_vol_difficulty"].view(-1).tolist()
            time_feats = self.data["time_features"].view(-1).tolist()
            self.state = (
                [self.previous_state[0]]  # balance
                + last_price  # stock prices initial
                # stocks on hold
                + self.previous_state[HOLDING_IDX:EMB_IDX]
                + emb_data  # tweet features
                + len_data  # tweet len
                + target_price  # target price
                + price_diff
                + vol_diff
                + text_diff
                + price_text_diff
                + all_diff
                + time_feats
            )

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]