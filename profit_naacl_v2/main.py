import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import pickle
from env import StockEnvTrade
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from tqdm import tqdm
import os
import datetime
import sys
from configs_stock import *
import re
import configs_stock

def train(
    num_iterations,
    agent,
    env_train,
    env_test,
    evaluate,
    validate_steps,
    output,
    max_episode_length=None,
    debug=False,
):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.0
    best_sharpe = -99999
    best_reward = -99999
    best_sortino = -99999
    best_calmar = -99999
    best_mdd = -99999
    best_cum_returns = -99999

    observation = None
    pbar = tqdm(total=num_iterations)
    
    print("args.warmup ", args.warmup) #25
    global global_df 
    
    while step < num_iterations:
        print()
        print("step:", step)
        # new_row = pd.DataFrame({'step': [step]})
        # global_df = pd.concat([global_df, new_row], axis=0, ignore_index=True)
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env_train.reset())
            agent.reset(observation)

        if step == args.warmup+1:
            print("step == args.warmup+1")
        
        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env_train.step(action) #infoにsharpとかの情報が入る　この一行で行う取引は1日分（1月8日の全銘柄に対する取引）observationは元のpickleの第n要素　observation_2は第n+1要素(次の日)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length - 1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup:
            agent.update_policy()
            
        #savemodel_stepsごとにモデル保存
        savemodel_steps = 5440
        if step % savemodel_steps == 0 and step!=0:
            print("save model step", step)
            agent.save_model_bysteps(output, step)

        # [optional] evaluate ここ消すとモデル保存されなくね
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0: #and step != 0は自前
            print("VALIDATING!!! step = ", step)
            agent.is_training = False
            IS_TRAINING = False
            agent.eval()
            policy = lambda x: agent.select_action(x, decay_epsilon=False)

            (
                validate_reward,
                validate_sharpe,
                validate_sortino,
                validate_calmar,
                validate_mdd,
                validate_cum_returns,
            ) = evaluate(env_test, policy, debug=debug, visualize=True, save=False) #ここでtest_pickle使用

            agent.train()
            agent.is_training = True
            IS_TRAINING = True

            print(
                "[Evaluate] Step_{:07d}: mean_reward:{} mean_sharpe:{} mean_sortino:{} mean_calmar:{} mean_mdd:{} mean_cum_returns:{}".format(
                    step,
                    validate_reward,
                    validate_sharpe,
                    validate_sortino,
                    validate_calmar,
                    validate_mdd,
                    validate_cum_returns,
                )
            )
            if validate_sharpe > best_sharpe:
            # if validate_reward > best_reward: #修正
                best_sharpe = validate_sharpe
                best_reward = validate_reward
                best_sortino = validate_sortino
                best_calmar = validate_calmar
                best_mdd = validate_mdd
                best_cum_returns = validate_cum_returns
                best_step = step
                
                # if step > 50: #追加　テストでいちいち保存させないため
                #     print("saving model!!!!")
                #     agent.save_model(output, step)
                #     if not os.path.exists(os.path.join(output, "validate_reward")):
                #         os.makedirs(os.path.join(output, "validate_reward"))
                #     evaluate.save_results(os.path.join(output, "validate_reward"))

            print(
                "[BEST] Step_{:07d}: mean_reward:{} mean_sharpe:{} mean_sortino:{} mean_calmar:{} mean_mdd:{} mean_cum_returns:{}".format(
                    best_step, #元々stepだったがそれだと[BEST] Step_が常に最新のものになるのでおかしい
                    best_reward,
                    best_sharpe,
                    best_sortino,
                    best_calmar,
                    best_mdd,
                    best_cum_returns,
                )
            )
            # print("output:", output)
            print("eval_end")
            
        # sys.exit() #一巡で終了
        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)
        print("step_end:", step)

        if done:  # end of episode
            if debug:
                prGreen(
                    "#{}: episode_reward:{} steps:{}".format(
                        episode, episode_reward, step
                    )
                )

            agent.memory.append(
                observation, agent.select_action(observation), 0.0, False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.0
            episode += 1
        pbar.update(1)

def test(num_episodes, agent, env, evaluate, model_path, step, visualize=True, debug=False,):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        # validate_reward = evaluate(
        #     env, policy, debug=debug, visualize=visualize, save=False
        # )
        
        
        (validate_reward,validate_sharpe,validate_sortino, validate_calmar, validate_mdd, validate_cum_returns,
            ) = evaluate(env_test, policy, debug=debug, visualize=True, save=False)
         

        print("[Evaluate] Step_{}: mean_reward:{} mean_sharpe:{} mean_sortino:{} mean_calmar:{} mean_mdd:{} mean_cum_returns:{}".format(
                    step,
                    validate_reward,
                    validate_sharpe,
                    validate_sortino,
                    validate_calmar,
                    validate_mdd,
                    validate_cum_returns,
                )
            )
        
        
        
        
        if debug:
            prYellow("[Evaluate] #{}: mean_reward:{}".format(i, validate_reward))

if __name__ == "__main__": #実行時の引数

    parser = argparse.ArgumentParser(description="PyTorch on TORCS with Multi-modal")

    parser.add_argument(
        "--mode", default="train", type=str, help="support option: train/test"
    )
    parser.add_argument(
        "--env", default="Humanoid-v2", type=str, help="open-ai gym environment"
    )
    parser.add_argument(
        "--hidden1",
        default=400,
        type=int,
        help="hidden num of first fully connect layer",
    )
    parser.add_argument(
        "--hidden2",
        default=300,
        type=int,
        help="hidden num of second fully connect layer",
    )
    parser.add_argument("--rate", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--prate",
        default=0.0001,
        type=float,
        help="policy net learning rate (only for DDPG)",
    )
    parser.add_argument(
        "--warmup",
        default=25,
        type=int,
        help="time without training but only filling the replay memory",
    )
    parser.add_argument("--discount", default=0.99, type=float, help="")
    parser.add_argument("--bsize", default=1, type=int, help="minibatch size")
    parser.add_argument("--rmsize", default=25, type=int, help="memory size")
    parser.add_argument("--window_length", default=1, type=int, help="")
    parser.add_argument(
        "--tau", default=0.001, type=float, help="moving average for target network"
    )
    parser.add_argument("--ou_theta", default=0.15, type=float, help="noise theta")
    parser.add_argument("--ou_sigma", default=0.2, type=float, help="noise sigma")
    parser.add_argument("--ou_mu", default=0.0, type=float, help="noise mu")
    parser.add_argument(
        "--validate_episodes",
        default=1,
        type=int,
        help="how many episode to perform during validate experiment",
    )
    parser.add_argument("--max_episode_length", default=500, type=int, help="")
    parser.add_argument(
        "--validate_steps",
        default=25,
        type=int,
        help="how many steps to perform a validate experiment",
    )
    parser.add_argument("--output", default="output_m1", type=str, help="")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--init_w", default=0.003, type=float, help="")
    parser.add_argument(
        "--train_iter", default=200000, type=int, help="train iters each timestep" #学習回数
    )
    parser.add_argument(
        "--epsilon", default=50000, type=int, help="linear decay of exploration policy"
    )
    parser.add_argument("--seed", default=-1, type=int, help="")
    parser.add_argument(
        "--resume", default="default", type=str, help="Resuming model path for testing"
    )
    parser.add_argument(
        "--model", default="time", type=str, help="which model to choose <profit, simple, plain>"
    )
    parser.add_argument(
        "--diff",
        default=None,
        type=str,
        help="diff for reward <price, vol, text, price_text, pvt> default: (None)",
    )
    
    parser.add_argument(
        "--test_pickle", default=None, type=str, help="which model to choose <profit, simple, plain>"
    )
    
    parser.add_argument(
        "--train_pickle", default=None, type=str, help="which model to choose <profit, simple, plain>"
    )
    
    parser.add_argument(
        "--initial_account_balance", default=None, type=str, help="which model to choose <profit, simple, plain>"
    )
    
    parser.add_argument(
        "--resumes_folder_path", default=None, type=str, help="which model to choose <profit, simple, plain>"
    )
    
    parser.add_argument(
        "--wantoutput", default="no", type=str, help="モデルのoutputが欲しいか"
    )

    args = parser.parse_args()
    
    current_datetime = datetime.datetime.now()
    print("start time:", current_datetime)
    print("INPUT_TEXT:", INPUT_TEXT)
    print("USING_MODEL:", USING_MODEL)
    
    wantoutput = args.wantoutput
    if wantoutput=="no":
        print("output is not made")
        output ="hoge"
        
    else:
        output = get_output_folder(args.output, args.env)
        print("output: ", output) #output/Humanoid-v2-run22
    
    
    if args.resume == "default":
        resume = "output/{}-run0".format(args.env) #これテスト時のモデル
        
    # resume = "output/Humanoid-v2-run4" #テスト時のモデル
        
    train_pickle = args.train_pickle #"/home/ubuntu/datasets/pickles_oneSECtimefeature_0tumeru/S_test_data_sorted_in_jsons_6_2015-01-01~2015-09-30_20_7_5_384.pkl"
    test_pickle = args.test_pickle
    print("train_pickle: ", train_pickle)
    print("test_pickle: ", test_pickle)
    # print("TRANSACTION_FEE_PERCENT: ", 0)
    
    # sys.exit()

    with open(train_pickle, "rb") as f:
        data_train = pickle.load(f)

    with open(test_pickle, "rb") as f:
        data_test = pickle.load(f)

    env_train = StockEnvTrade(data_train, 0, args)
    env_test = StockEnvTrade(data_test, 0, args)

    nb_states = env_train.observation_space.shape[0]
    nb_actions = env_train.action_space.shape[0]

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(
        args.validate_episodes,
        args.validate_steps,
        output,
        max_episode_length=len(data_test),
    )
    
    # configs_stock.INITIAL_ACCOUNT_BALANCE = 100000

    print("args.validate_steps:", args.validate_steps)
    print("args.epsilon:", args.epsilon)
    print("HMAX_NORMALIZE:", HMAX_NORMALIZE)
    print("SELECT_ACTION:", SELECT_ACTION)
    print("TRANSACTION_FEE_PERCENT", TRANSACTION_FEE_PERCENT)
    print("args.train_iter:", args.train_iter)
    print("args.seed:", args.seed)
    print("args.bsize:", args.bsize)
    args_dict = args.__dict__  # 引数の値を辞書型で取得
    print(args_dict)
    print("""MEMO:
          """)

    if args.mode == "train":
        train(
            args.train_iter,
            agent,
            env_train,
            env_test,
            None, #None evaluate
            args.validate_steps,
            output,
            max_episode_length=len(data_train),
            debug=args.debug,
        )

    elif args.mode == "test":
        print("this is test mode")
        #resumeの複数入っているフォルダの場合
        resumes_folder_path = args.resumes_folder_path
        resumes_folder = os.listdir(resumes_folder_path)
        resumes_folder.sort(key=lambda x: os.path.getmtime(os.path.join(resumes_folder_path, x)))
        print(resumes_folder)
        contains_pkl = any(file.endswith('.pkl') for file in resumes_folder)
        # sys.exit()
        # folder_names = [name for name in sorted(os.listdir(resumes_folder)) if os.path.isdir(os.path.join(resumes_folder, name))]
        if contains_pkl==False:
            for resume in resumes_folder:
                if(re.findall(r'\d+', resume)):
                    step = re.findall(r'\d+', resume)[0]
                    # if int(step) > 27000: #step27200のテストのみ実施
                    print("step", step)
                    resume = os.path.join(resumes_folder_path, resume)
                    print("test model:", resume)
                    # test(
                    #     args.validate_episodes,
                    #     agent,
                    #     env_test,
                    #     evaluate,
                    #     resume,
                    #     step,
                    #     visualize=True,
                    #     debug=args.debug,
                    # )
            # except:
            #     pass
            # break
        else:
            print("stepは１種類のみ")
            step = re.findall(r'\d+', resumes_folder_path)[-1]
            print("step", step)
            resume = resumes_folder_path
            print("test model:", resume)
            test(
                        args.validate_episodes,
                        agent,
                        env_test,
                        evaluate,
                        resume,
                        step,
                        visualize=True,
                        debug=args.debug,
                    )

    else:
        raise RuntimeError("undefined mode {}".format(args.mode))
    
    
    print("all finish")
    # print(global_df)
