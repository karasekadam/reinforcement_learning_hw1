from infrastructure.envs.tabular_wrapper import EnvWrapper
from infrastructure.utils.logger import Logger

from datetime import datetime
import gymnasium as gym
import numpy as np
import pandas as pd

"""
    The following classes define the interface we use to implement and evaluate
    RL algorithms in this homework.

    You will implement the RL algorithms as a subclasses of the common Trainer 
    class. The only two required methods are `__init__` and `train`.

    Each `train()` method should perform the training and return the specified
    Policy object. This is to enable automatic evaluation on our side, and to 
    allow you to use the provided visualization utilities.
"""

class Policy:
    def __init__(self, **kwargs):
        raise NotImplementedError()

    # Should sample an action from the policy in the given state
    def play(self, state : int, greedy=False) -> int:
        raise NotImplementedError()

    # Raw output of the policy, this could later be logits/etc.
    # However, for this homework the output of raw(state) MUST be
    # the estimated value of the given state under the policy
    def raw(self, state : int) -> float:
        raise NotImplementedError()



"""
    These are the policy objects you will work with.
    Check the return types of each `train()` method.
"""
class ValuePolicy(Policy):
    def __init__(self, values, decisions):
        self.values = values
        self.decisions = decisions

    def play(self, state, greedy=False):
        return self.decisions[state]

    def raw(self, state):
        return self.values[state]


class GreedyPolicy(Policy):
    def __init__(self, q_table):
        self.q_table = q_table

    def play(self, state, greedy=False):
        return np.argmax(self.q_table[state])

    def raw(self, state):
        return np.max(self.q_table[state])



class EpsGreedyPolicy(Policy):
    def __init__(self, q_table, eps):
        self.q_table = q_table
        self.eps = eps

    # The greedy flag is used by our rendering utilities
    def play(self, state, greedy=False):
        if not greedy and (np.random.rand() < self.eps):
            return np.random.choice(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state])

    def raw(self, state):
        return np.max(self.q_table[state])


class Trainer:
    # Stores the EnvWrapper object
    def __init__(self, env : EnvWrapper, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the number of iterations for VI, or total number of calls to env.step() for QL, SARSA, and MC
    def train(self, gamma : float, steps : int, **kwargs) -> Policy:
        raise NotImplementedError()

"""
    VALUE ITERATION
"""

class VITrainer(Trainer):

    def __init__(self, env, **kwargs):
        # `env` is saved as `self.env`
        super(VITrainer, self).__init__(env)

    def train(self, gamma, steps, **kwargs) -> ValuePolicy:
        # TODO - complete the Value Iteration Algorithm, 
        # and execute steps number of iterations

        # The states are numbers \in [0, ... nS-1], same with actions.
        nS = self.env.num_states()
        nA = self.env.num_actions()
        values = ...

        # recall that environment dynamics are available as full tensors:
        # w. `self.env.get_dynamics_tensor()`, or via `get_transition(s, a, s')`

        # Make sure you return an object extending the Policy interface, so
        # that you are able to render the policy and we can evaluate it.
        pass

"""
    Q-LEARNING
"""

class QLTrainer(Trainer):

    def __init__(self, env, **kwargs):
        super(QLTrainer, self).__init__(env)
        # feel free to add stuff here as well

    def train(self, gamma, steps, eps, lr, explore_starts=False, logger=None, **kwargs) -> GreedyPolicy:
        # TODO - complete the QLearning algorithm that uses the supplied
        # values of eps/lr (for the whole training). Use an epsilon-greedy exploration policy.

        step = 0

        # TODO: modify this call for exploring starts as well
        state, info = self.env.reset()
        done = False
        
        while not done and step < steps:
            # TODO: action selection
            action = np.random.randint(4)

            succ, rew, terminated, truncated, _ = self.env.step(action)

            step += 1

            # TODO: update values
            if terminated or truncated:
                done = True
            
            # TODO: Report data through the provided logger
            if logger is not None:
                # TODO: Evaluate policy, average per episode reward, etc.
                    logger.write({"rew": rew, "termination": terminated}, step)


        # TODO: remember to only perform `steps` samples from the training environment


"""
    SARSA
"""

class SARSATrainer(Trainer):

    def __init__(self, env, **kwargs):
        super(SARSATrainer, self).__init__(env)

    def train(self, gamma, steps, eps, lr, explore_starts=False, **kwargs) -> EpsGreedyPolicy:
        # TODO - complete the SARSA algorithm that uses the supplied values of
        # eps/lr and exploring starts.
        pass


"""
    EVERY VISIT MONTE CARLO CONTROL
"""

class MCTrainer(Trainer):
    def __init__(self, env, **kwargs):
        super(MCTrainer, self).__init__(env)

    def train(self, gamma, steps, eps, explore_starts=False, **kwargs) -> EpsGreedyPolicy:
        # TODO - Complete every visit MC-control, which uses an epsilon greedy
        # exploration policy
        pass

"""
    Evaluation
        
    As part of the exercise sheet, you are expected to deliver visualizations
    of the learning curves of each algorithm on each environment.

    To achieve this, we have prepared a `Logger` class in
    infrastructure/utils/logger.py, which you can use to easily log data into
    csv files. You can see demonstration of the logger interface in `QLTrainer::train()`, 
    which is called in the main function.
"""



"""
    We will demonstrate the rendering methods implemented
    in the wrapper using a dummy policy.
"""
class RandomPolicy(Policy):
    """
        A dummy policy that returns random actions and random values
    """
    def __init__(self, nA):
        self.nA = nA

    def play(self, state, greedy=False):
        return np.random.randint(self.nA)

    def raw(self, state):
        return np.random.randint(42)


if __name__ == "__main__":

    current_timestamp = datetime.now()
    timestamp = current_timestamp.strftime('%Y-%m-%d-%H:%M:%S')

    """
        These are the three environments that we will use during this
        assignment. The number of samples (`step()` calls) you can
        make in these is unlimited. This is to make it easier for you to test
        your solution and to generate your report.
    """

    FrozenLake = EnvWrapper(gym.make('FrozenLake-v1', map_name='4x4'))
    LargeLake = EnvWrapper(gym.make('FrozenLake-v1', map_name='8x8'))
    CliffWalking = EnvWrapper(gym.make('CliffWalking-v0'))

    """
        However, in the automatic evaluation we will also check the number of
        samples you take in the environment. 

        For example, calling `QLTrainer.train()` with `steps=10` should 
        only sample from the environment 10 times. You can check that your
        implementation does not sample the environment too many times by
        setting `max_samples=n` in the Wrapper constructor; see below:

    """
    LimitedEnv = EnvWrapper(gym.make('CliffWalking-v0'), max_samples = 10)



    """
        Logging example - walk through the CliffWalking environment
        randomly and log reward collected each step to the directory below.

        The logs should be located at results/test/logs.csv. 
    """
    log_dir = "results/test/"
    logger = Logger(log_dir)
    QLTrainer(CliffWalking).train(gamma=1.0, steps=42, eps=0.42, lr=0.42, logger=logger)
    df = pd.read_csv(log_dir + "logs.csv", sep=";")
    print(df.head(10))



    """ 
        You can also use the `render_mode="human"` argument for Gymnasium to
        see an animation of your agent's decisions.
    """
    AnimatedEnv = EnvWrapper(gym.make('FrozenLake-v1', map_name='4x4'
                                                     , render_mode='human'),
                             max_samples = -1)
    AnimatedEnv.reset()
    # Walk around randomly for a bit
    for i in range(10):
        obs, rew, done, trunc, _ = AnimatedEnv.step(np.random.randint(4))
        if done:
            AnimatedEnv.reset()



    """
        Rendering example - using env.render_policy() to get a value heatmap as
        well as the greedy actions w.r.t. the policy values.
    """
    def render_random(env):
        """
            Plots heatmap of the state values and arrows corresponding to actions on `env`
        """
        env.reset(randomize=False)
        policy = RandomPolicy(env.num_actions())
        env.render_policy(policy, label= "RandomPolicy")

    render_random(FrozenLake)
