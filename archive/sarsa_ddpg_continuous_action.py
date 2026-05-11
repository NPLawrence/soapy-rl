# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from neuromancer.system import Node, System
from neuromancer.modules import blocks
from neuromancer.dataset import DictDataset
from neuromancer.constraint import variable, Objective
from neuromancer.loss import PenaltyLoss, AggregateLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer
from neuromancer.plot import pltCL, pltPhase, plot_trajectories
import neuromancer.psl as psl
from neuromancer.dynamics import ode, integrators
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.dynamics.ode import ODESystem

import matplotlib.pyplot as plt

from pendulum import DoubleInvertedPendulum
from soap import SOAP

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "dip"
    """the environment id of the Atari game"""
    total_epochs: int = 500
    """total epochs"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""

    optimizer: str = "soap"
    loss: str = "gc"
    num_data: int = 1000000



if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    nx, nu = 6, 1

    ts = 0.05

    # white-box ODE model with no-plant model mismatch
    gt_ode = DoubleInvertedPendulum()                   # ODE system equations implemented in PyTorch

    # integrate continuous-time ODE
    integrator = integrators.RK4_Trap(gt_ode, h=torch.tensor(ts))   # RK4, RK4_Trap, Runge_Kutta_Fehlberg, LeapFrog

    dynamics = Node(integrator, ['X', 'U'], ['X_next'], name='model')

    X_shift = Node(lambda x: x, ['X_next'], ['X'], name='x_shift')

    observation = Node(lambda x: 1.0 - torch.cos(x[:,1:3]), ['X_next'], ['Y'])

    if args.loss == 'gc':
        prob = lambda x: torch.exp(-0.5*(x**2).sum(axis=1, keepdims=True) / 1.0**2)
    elif args.loss == 'l2':
        prob = lambda x: 0.5*(x**2).sum(axis=1, keepdims=True)
    cost = Node(prob, ['Y'], ['l'])

    

    actor = blocks.MLP_bounds(nx, nu, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.GELU,
                    min = -10.0, max = 10.0,
                    hsizes=[256, 256]) # [256, 256] & GELU is nice and smooth

    qf1 = blocks.MLP_bounds(nx, 1, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.GELU,
                    min = 0.0, max = 1 / (1 - args.gamma),
                    hsizes=[256, 256]) # [256, 256] & GELU is nice and smooth

    policy = Node(actor, ['X'], ['U'], name='policy')
    # next_policy = Node(actor, ['X_next'], ['U'], name='policy')
    value = Node(lambda x: -qf1(x, actor(x)), ['X'], ['V'], name='value')
    # q_value = Node(lambda x, u: -qf1(x, u), ['X', 'U'], ['Q'], name='q_value')
    td_target = Node(lambda x, l: l + args.gamma*qf1(x), ['X_next', 'l'], ['target_V'])
    # Q_value = Node(lambda x, l: l + args.gamma*qf1(x), ['X_next'], ['Q'])

    # td_error = lambda x, y: 0.5*((qf1(x, actor(x))-y.detach())**2).sum(axis=1, keepdims=True)
    td_error = lambda x, y: 0.5*((qf1(x)-y)**2).sum(axis=1, keepdims=True)

    # td_target = Node(lambda x, l: l + args.gamma*qf1(x, actor(x)), ['X', 'l'], ['target_V'])

    td_cost = Node(td_error, ['X', 'target_V'], ['critic_cost'])

    # closed loop system definition
    cl_system = System([policy, dynamics, observation, cost, td_target, td_cost], nsteps=1)
    cl_system_eval = System([policy, dynamics, observation, cost, td_target, td_cost, X_shift], nsteps=100)

    #### Insert sketchy optimizer here!! ####
    if args.optimizer == "adam":
        q_optimizer = optim.AdamW(qf1.parameters())#, lr=args.learning_rate)
        actor_optimizer = optim.AdamW(policy.parameters())#, lr=args.learning_rate)
    elif args.optimizer == "soap":
        q_optimizer = SOAP(list(qf1.parameters()))#, lr=args.learning_rate)
        actor_optimizer = SOAP(list(policy.parameters()))#, lr=args.learning_rate) 
        # optimizer = SOAP(list(qf1.parameters()) + list(policy.parameters()))     


    # Define policy optimization problem
    u = variable('U')
    x = variable('X')

    # pos = variable('pos')
    pos_max = 5
    state_lower_bound_penalty = 1.*(x[0] > -1*pos_max)
    state_upper_bound_penalty = 1.*(x[0] < pos_max)

    lpred = variable('target_V')

    l_loss = Objective(var=lpred, name='stage_loss')

    # if constrained_pos:
    constraints = [state_lower_bound_penalty, state_upper_bound_penalty]
    # else:
        # constraints = []

    # if loss == 'gc':
    #     obj = LogLoss([l_loss], constraints) # state_lower_bound_penalty, state_upper_bound_penalty
    # elif loss == 'l2':
        
    policy_obj = PenaltyLoss([-1.0*l_loss], []) # state_lower_bound_penalty, state_upper_bound_penalty


    policy_problem = Problem([cl_system], policy_obj)

    eval_problem = Problem([cl_system_eval], PenaltyLoss([Objective(var=variable('l'))], []))

    td_var = variable('critic_cost')

    value_loss = Objective(var=td_var)
    value_obj = PenaltyLoss([value_loss], []) # state_lower_bound_penalty, state_upper_bound_penalty
    critic_problem = Problem([cl_system], value_obj)

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    # init_data = torch.rand((args.num_data, 1, nx))*10.0 - 5.0
    # init_data[:,:,[1,2,4,5]] *= 2.0

    # rand_actions = torch.rand((args.num_data, 1, nu))*20.0 - 10.0

    # data = DictDataset({'X': init_data}, name='train') # Split conditions into train and dev

    init_dev_data = torch.rand((10000, 1, nx))*10.0 - 5.0
    init_dev_data[:,:,[1,2,4,5]] *= 2.0

    rand_actions_dev = torch.rand((10000, 1, nu))*20.0 - 10.0

    dev_data = DictDataset({'X': init_dev_data}, name='dev') # Split conditions into train and dev

    eval_init = torch.zeros((1000, 1, nx))
    eval_init[:,:,1:3] = torch.pi #* (torch.rand_like(td[0:split,:,1:3])*0.2 + 0.9)
    eval_data = {'X': 0.1*torch.randn((1000, 1, nx)) + eval_init}

    # traj_data = cl_system(data)

    # train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
    #                                         collate_fn=data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
    
    for epoch in range(args.total_epochs):
        # ALGO LOGIC: put action logic here

        print("EPOCH ", epoch)
        init_data = torch.rand((args.num_data, 1, nx))*10.0 - 5.0
        init_data[:,:,[1,2,4,5]] *= 2.0

        # rand_actions = torch.rand((args.num_data, 1, nu))*20.0 - 10.0

        data = DictDataset({'X': init_data}, name='train') # Split conditions into train and dev
        train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                                collate_fn=data.collate_fn, shuffle=True)
    
        critic_trainer = Trainer(
            critic_problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer=q_optimizer,
            epochs=10,
            train_metric="train_loss",
            dev_metric="dev_loss",
            eval_metric='dev_loss',
            warmup=0,
        )
        print("Critic training...")
        critic_trainer.train()

        policy_trainer = Trainer(
            policy_problem,
            train_loader,
            dev_loader,
            dev_loader,
            optimizer=actor_optimizer,
            epochs=1,
            train_metric="train_loss",
            dev_metric="dev_loss",
            eval_metric='dev_loss',
            warmup=0,
        )

        print("Actor training...")
        policy_trainer.train()

        with torch.no_grad():
            traj_data = cl_system_eval(eval_data)
            eval_loss = eval_problem.loss(traj_data)['loss'].detach().numpy()

            fig, axes = plt.subplots(2)
            # print(traj_data['Y'].shape)
            axes[0].plot(traj_data['Y'][0:10,:,0].detach().reshape(100, -1).numpy())
            axes[1].plot(traj_data['Y'][0:10,:,1].detach().reshape(100, -1).numpy())
            wandb.log({"plots/dip_plot": wandb.Image(fig)})
            plt.close()

            cl_system_eval.nsteps = 1
            network_eval_data = cl_system_eval({'X': init_dev_data})
            q_loss = network_eval_data['critic_cost'].detach().numpy().mean()
            cl_system_eval.nsteps = 100

            print("Eval performance ", eval_loss)
            print("Q eval ", q_loss)

        writer.add_scalar("losses/eval_loss", eval_loss, epoch)
        writer.add_scalar("losses/eval_performance", q_loss, epoch)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    if args.track:
        wandb.finish()
    writer.close()
