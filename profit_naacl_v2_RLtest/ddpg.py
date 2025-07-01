import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from util import *
from configs_stock import *
import random

# from model import Actor, Critic #24行目　self.actor = Actor() 動かないので追加　それでもダメ

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):

        if args.seed > 0:
            self.seed(args.seed)
        # if args.model == "profit": 
        #     from model import Actor, Critic
        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        if USING_MODEL=="model":
            from model import Actor, Critic #下のエラー回避
        
        elif USING_MODEL=="model_2":
            from model_2 import Actor, Critic
        
        elif USING_MODEL=="model_3":
            from model_3 import Actor, Critic
            
        elif USING_MODEL=="model_4":
            from model_4 import Actor, Critic
            
        elif USING_MODEL=="model_5":
            from model_5 import Actor, Critic
            
        self.actor = Actor() #UnboundLocalError: local variable 'Actor' referenced before assignment
        self.actor_target = Actor()
        self.actor_optim = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(
            self.actor_target, self.actor
        )  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(
            limit=args.rmsize, window_length=args.window_length
        )
        self.random_process = OrnsteinUhlenbeckProcess(
            size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma
        )

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        #
        if USE_CUDA:
            self.cuda()

    def update_policy(self):
        # Sample batch
        (
            state_batch,
            action_batch,
            reward_batch, #ここで報酬関数が使われている
            next_state_batch,
            terminal_batch,
        ) = self.memory.sample_and_split(self.batch_size)

        print("next_state_batch", next_state_batch)
        temp = to_tensor(next_state_batch, volatile=True)
        print("to_tensor(next_state_batch, volatile=True)", temp.shape)
        # print( state_batch,
        #     action_batch,
        #     reward_batch, #ここで報酬関数が使われている
        #     next_state_batch,
        #     terminal_batch,)
        # Prepare for the target q batch
        next_q_values = self.critic_target(
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        )
        
        #next_q_values.volatile = False　userwarningの原因

        with torch.no_grad():#next_q_values.volatile = Falseの書きかえ
            next_q_values = next_q_values.detach()

        target_q_batch = (
            to_tensor(reward_batch)
            + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values
        )

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic(to_tensor(state_batch), to_tensor(action_batch))
        # ("q_batch:",  q_batch)
        print("target_q_batch:",  target_q_batch)
        value_loss = criterion(q_batch, target_q_batch) #0に固定されてる？
        value_loss.backward()
        print("value_loss:", value_loss)
        # new_row = pd.DataFrame({'value_loss': [value_loss]})
        # global_df = pd.concat([global_df, new_row], axis=0, ignore_index=True)
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) #0214
        self.critic_optim.step()

        # # Actor update
        # self.actor.zero_grad()

        # # print()
        # policy_loss = -self.critic(
        #     to_tensor(state_batch), self.actor(to_tensor(state_batch))
        # )

        # print("policy_loss:", policy_loss)
        # policy_loss = policy_loss.mean()
        # policy_loss.backward()
        # print("policy_loss_mean:", policy_loss)
        # # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) #0214
        # self.actor_optim.step()
        
        # Actorの更新（自然勾配法を使用）
        self.actor.zero_grad()
        policy_loss = -self.critic(
            to_tensor(state_batch), self.actor(to_tensor(state_batch))
        ).mean()
        policy_loss.backward()

        # フィッシャー情報行列を計算
        fisher_matrix = self.compute_fisher_matrix(state_batch)
        natural_gradient = self.compute_natural_gradient(fisher_matrix)

        # 自然勾配を用いてActorを更新
        for param, grad in zip(self.actor.parameters(), natural_gradient):
            param.data -= self.actor_optim.param_groups[0]['lr'] * grad

        # ターゲットネットワークの更新
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def compute_fisher_matrix(self, state_batch):
        actions = self.actor(to_tensor(state_batch))
        log_probs = torch.log(actions + 1e-8)
        fisher_matrix = torch.mm(log_probs.T, log_probs) / len(state_batch)
        return fisher_matrix

    def compute_natural_gradient(self, fisher_matrix):
        """自然勾配を計算"""
        # fisher_inv = torch.inverse(fisher_matrix + 1e-8 * torch.eye(fisher_matrix.size(0)))
        fisher_inv = torch.inverse(fisher_matrix + 1e-8 * torch.eye(fisher_matrix.size(0), device=fisher_matrix.device))
        gradients = [param.grad.view(-1) for param in self.actor.parameters()]
        gradients = torch.cat(gradients)
        natural_gradient = torch.mm(fisher_inv, gradients.unsqueeze(1)).squeeze(1)
        return natural_gradient.split([param.numel() for param in self.actor.parameters()])


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            # print("ddpg_observe_self.memory", self.memory)
            self.s_t = s_t1

    def random_action(self):
        action = np.random.uniform(-1.0, 1.0, self.nb_actions)
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0) #値は1か-1
        print("action1:", action, type(action))
        # print(self.is_training, self.epsilon, self.random_process.sample())
        
        if self.is_training:
            if SELECT_ACTION == "default":
                action += self.is_training * max(self.epsilon, 0) * self.random_process.sample() #ノイズが混じる　あまり大きくない？
                # print("self.random_process.sample()", self.random_process.sample())
                print("action2:", action)
                action = np.clip(action, -1.0, 1.0) #actionの値の範囲を-1~1にする
                print("action3:", action)
                
            if SELECT_ACTION == "random": #一定の確率でactionのランダムな箇所の値を乱数にする
                random_value = 2 * random.random()
                # print(random_value, self.epsilon)
                if random_value < self.epsilon: #self.epsilonは初期値１なので初めは５割で分岐　そこから減ってく
                    # print("分岐成功")
                    random_int = random.randint(0, 19)
                    # print(random_int)
                    # print(type(action[random_int]), type(random.uniform(-1, 1)))
                    action[random_int] = random.uniform(-1, 1)
                    print("action2:", action)
                    
            if SELECT_ACTION == "plus-minus-change": #一定の確率でactionのランダムな箇所の正負を入れ替える
                random_value = 2 * random.random()
                # print(random_value, self.epsilon)
                if random_value < self.epsilon: #self.epsilonは初期値１なので初めは５割で分岐　そこから減ってく
                    # print("分岐成功")
                    random_int = random.randint(0, 19)
                    # print(random_int)
                    # print(type(action[random_int]), type(random.uniform(-1, 1)))
                    action[random_int] = -action[random_int]
                    print("action2:", action)
        
        if decay_epsilon: #decay_epsilon フラグが真（True）の場合、ε の値を徐々に減少させ、最適な行動をより頻繁に選択するように調整することができます
            self.epsilon -= self.depsilon #１ステップごとにself.epsilon が1/50000ずつ減少
            
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None:
            return

        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))
        self.actor_target.load_state_dict(
            torch.load("{}/actor_target.pkl".format(output))
        )
        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output)))
        self.critic_target.load_state_dict(
            torch.load("{}/critic_target.pkl".format(output))
        )

    def save_model(self, output, step):
        if(step<50):
            torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
            torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))
            torch.save(self.actor_target.state_dict(), "{}/actor_target.pkl".format(output))
            torch.save(
                self.critic_target.state_dict(), "{}/critic_target.pkl".format(output)
            )
        
        else: #stepが50以上の時モデルを新規保存
            try:
                actor_folder = "{}/step_" + step + "/actor.pkl"
                critic_folder = "{}/step_" + step + "/critic.pkl"
                actor_target_folder = "{}/step_" + step + "/actor_target.pkl"
                critic_target_folder = "{}/step_" + step + "/critic_target.pkl"
                torch.save(self.actor.state_dict(), actor_folder.format(output))
                torch.save(self.critic.state_dict(), critic_folder.format(output))
                torch.save(self.actor_target.state_dict(), actor_target_folder.format(output))
                torch.save(
                    self.critic_target.state_dict(),critic_target_folder.format(output)
                )
            except:
                torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
                torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))
                torch.save(self.actor_target.state_dict(), "{}/actor_target.pkl".format(output))
                torch.save(
                self.critic_target.state_dict(), "{}/critic_target.pkl".format(output)
                )

    #1000stepごとにこれを呼び出してモデルを保存
    def save_model_bysteps(self, output, step):
        folder_path = "{}/".format(output) + "step_" + str(step)
        os.mkdir(folder_path)
        print("saving model to ", folder_path)
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(folder_path))
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(folder_path))
        torch.save(self.actor_target.state_dict(), "{}/actor_target.pkl".format(folder_path))
        torch.save(
            self.critic_target.state_dict(), "{}/critic_target.pkl".format(folder_path)
        )

    def seed(self, s):
        # print("seed s:", s)
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)