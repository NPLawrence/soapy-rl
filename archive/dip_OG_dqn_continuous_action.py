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
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
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
from pendulum import InvertedPendulum
from pendulum import Acrobot


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
    env_id: str = "pendulum"
    """the environment id"""
    total_epochs: int = 200
    """total epochs"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    batch_size: int = 1024
    """the batch size of sample from the reply memory"""

    optimizer: str = "adamw"
    loss: str = "gc"
    num_data: int = 10000000
    critic_loss: str = "abs"

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
    nx, nu = 4, 1
    
    # actions = torch.tensor([-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0])
    actions = 1*torch.linspace(-1.0, 1.0, steps=11)
    # actions = 2*torch.cat((torch.logspace(-3.0, 0.0, steps=10),-1*torch.logspace(-3.0, 0.0, steps=10),torch.tensor([0.0])))
    na = actions.shape[0] # number of actions (different from action dimension)
    # print(actions) 
    ts = 0.05

    # white-box ODE model with no-plant model mismatch
    gt_ode = DoubleInvertedPendulum()                   # ODE system equations implemented in PyTorch
    # gt_ode = InvertedPendulum()
    gt_ode = Acrobot()

    # integrate continuous-time ODE
    integrator = integrators.RK4(gt_ode, h=torch.tensor(ts))   # RK4, RK4_Trap, Runge_Kutta_Fehlberg, LeapFrog

    dynamics = Node(integrator, ['X', 'U'], ['X_next'], name='model')

    X_shift = Node(lambda x: x, ['X_next'], ['X'], name='x_shift')

    observation = Node(lambda x: -torch.cos(x[:,[0]]) - torch.cos(x[:,[1]] + x[:,[0]]), ['X'], ['Y']) # - 0.5*x[:,[3]]**2
    cos_theta  = Node(lambda x: torch.cos(x[:,[0,1]]), ['X'], ['cos_theta'])

    if args.loss == 'gc':
        # prob = lambda x: torch.exp(-0.5*(x**2).sum(axis=1, keepdims=True) / 1.0**2)
        prob = lambda x: x
    elif args.loss == 'l2':
        prob = lambda x: 0.5*(x**2).sum(axis=1, keepdims=True)
    cost = Node(prob, ['Y'], ['l'])


    qf1 = blocks.MLP(nx, na, bias=True,
                    linear_map=torch.nn.Linear,
                    nonlin=torch.nn.Tanh,
                    hsizes=[64, 64]) 

    policy = Node(lambda x: torch.argmax(qf1(x), dim=-1, keepdim=True), ['X'], ['A'], name='policy')
    control_action = Node(lambda a: actions[a], ['A'], ['U'], name='control')
    q_value = Node(qf1, ['X'], ['Q'], name='q_value') 
    td_target = Node(lambda x, l: l + args.gamma*torch.max(qf1(x), dim=-1, keepdim=True)[0], ['X_next', 'l'], ['target_V'])

    if args.critic_loss == "mse":
        td_error = lambda x, a, y: 0.5*((qf1(x).gather(1, a)-y.detach())**2).sum(axis=1, keepdims=True)
    elif args.critic_loss == "abs":
        td_error = lambda x, a, y: 0.5*torch.abs(qf1(x).gather(1, a)-y.detach()).sum(axis=1, keepdims=True)
        # td_error = lambda x, a, y: 0.5*torch.nn.functional.huber_loss(qf1(x).gather(1, a), y.detach())

    td_cost = Node(td_error, ['X', 'A', 'target_V'], ['critic_cost'])

    # closed loop system definition
    cl_system = System([control_action, dynamics, observation, cost, cos_theta, td_target, td_cost], nsteps=1)
    cl_system_eval = System([policy, control_action, dynamics, observation, cost, cos_theta, td_target, td_cost, X_shift], nsteps=100)

    #### Insert  optimizer here!! ####
    if args.optimizer == "adam":
        q_optimizer = optim.Adam(qf1.parameters())#, lr=args.learning_rate)
    elif args.optimizer == "adamw":
        q_optimizer = optim.AdamW(qf1.parameters())#, lr=args.learning_rate)
    elif args.optimizer == "soap":
        q_optimizer = SOAP(list(qf1.parameters()))#, lr=args.learning_rate)
    elif args.optimizer == "rmsprop":
        q_optimizer = optim.RMSprop(qf1.parameters())#, lr=args.learning_rate)
    elif args.optimizer == "sgd":
        q_optimizer = optim.SGD(qf1.parameters())#, lr=args.learning_rate)
    elif args.optimizer == "asgd":
        q_optimizer = optim.ASGD(qf1.parameters())#, lr=args.learning_rate)
    
    # lr_scheduler = ReduceLROnPlateau(q_optimizer, mode="min", factor=0.5, patience=5)
    lr_scheduler = ExponentialLR(q_optimizer, gamma=0.95)

    # Define policy optimization problem
    u = variable('U')
    x = variable('X')

    pos_max = 5
    state_lower_bound_penalty = 1.*(x[0] > -1*pos_max)
    state_upper_bound_penalty = 1.*(x[0] < pos_max)

    constraints = [state_lower_bound_penalty, state_upper_bound_penalty]

    eval_problem = Problem([cl_system_eval], PenaltyLoss([Objective(var=variable('l'))], []))

    td_var = variable('critic_cost')

    value_loss = Objective(var=td_var)
    value_obj = PenaltyLoss([value_loss], []) # state_lower_bound_penalty, state_upper_bound_penalty
    critic_problem = Problem([cl_system], value_obj)

    start_time = time.time()

    init_data = torch.zeros((args.num_data, 1, nx)) # *14.0 - 7.0
    init_data[:,:,0] = torch.pi
    init_actions = torch.randint(0, na, (args.num_data, 1, 1))


    data = DictDataset({'X': 5.0*torch.randn((args.num_data, 1, nx)) + init_data, 'A': init_actions}, name='train') # Split conditions into train and dev
    train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
                                            collate_fn=data.collate_fn, shuffle=True)

    init_dev_data = init_data
    init_dev_actions = init_actions

    dev_data = DictDataset({'X': init_dev_data, 'A': init_dev_actions}, name='dev') # Split conditions into train and dev

    eval_init = torch.zeros((100, 1, nx))
    # eval_init[:,:,0] = torch.pi #* (torch.rand_like(td[0:split,:,1:3])*0.2 + 0.9)
    eval_data = {'X': 0.1*torch.randn((100, 1, nx)) +  eval_init}

    # train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size,
    #                                         collate_fn=data.collate_fn, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size,
                                            collate_fn=dev_data.collate_fn, shuffle=True)
    
    for epoch in range(args.total_epochs):
        # ALGO LOGIC: put action logic here

        print("EPOCH ", epoch)

        ## we manually implement the training loop rather than use the neuromancer Trainer module. the Trainer
        # has built-in features like validation error and patience that get in the way of our problem setting
        # as we may have a nondecreasing TD cost, but improved overall policy performance
        losses = []
        for t_batch in train_loader:
           

            q_optimizer.zero_grad()
            output = critic_problem(t_batch)
            output["train_loss"].backward()
            # torch.nn.utils.clip_grad_norm_(critic_problem.parameters(), 100.0)
            q_optimizer.step()
            losses.append(output["train_loss"].detach().numpy())

        lr_scheduler.step()

        writer.add_scalar("losses/q_loss", np.mean(losses), epoch)

        with torch.no_grad():
            traj_data = cl_system_eval(eval_data)
            eval_loss = eval_problem.loss(traj_data)['loss'].detach().numpy()

            fig, axes = plt.subplots(4, constrained_layout=True, sharex=True)
            axes[0].plot(traj_data['cos_theta'][0,:,:].detach().numpy())
            axes[1].plot(traj_data['Y'][0,:,:].detach().numpy())
            axes[2].plot(traj_data['U'][0,:,0].detach().numpy())
            axes[3].plot(traj_data['X'][0,:,[0,2]].detach().numpy())

            if args.track:
                wandb.log({"plots/dip_plot": wandb.Image(fig)}, epoch)
            else:
                plt.savefig(f"epoch.png")
            plt.close()

            cl_system_eval.nsteps = 1
            network_eval_data = cl_system_eval({'X': init_dev_data})
            q_loss_eval = network_eval_data['critic_cost'].detach().numpy().mean()
            cl_system_eval.nsteps = 100

            print("Eval performance ", eval_loss)
            print("Q eval ", q_loss_eval)

        writer.add_scalar("losses/eval_return", eval_loss, epoch)
        writer.add_scalar("losses/eval_error", q_loss_eval, epoch)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(qf1.state_dict(), model_path)
        print(f"model saved to {model_path}")

    if args.track:
        wandb.finish()
    writer.close() 