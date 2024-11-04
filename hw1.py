from more_itertools.more import sample

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
    def play(self, state: int, greedy=False) -> int:
        raise NotImplementedError()

    # Raw output of the policy, this could later be logits/etc.
    # However, for this homework the output of raw(state) MUST be
    # the estimated value of the given state under the policy
    def raw(self, state: int) -> float:
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
    def __init__(self, env: EnvWrapper, **kwargs):
        self.env = env

    # `gamma` is the discount factor
    # `steps` is the number of iterations for VI, or total number of calls to env.step() for QL, SARSA, and MC
    def train(self, gamma: float, steps: int, **kwargs) -> Policy:
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
        values = [0 for _ in range(nS)]

        for _ in range(steps):
            new_values = [0 for _ in range(nS)]
            for state in range(nS):
                max_value = -np.inf
                for action in range(nA):
                    value = 0
                    for next_state in range(nS):
                        next_state_probability = self.env.get_transition(state, action, next_state)
                        reward = self.env.get_reward(state, action, next_state)
                        value += (next_state_probability * (reward + gamma * values[next_state]))
                    max_value = max(max_value, value)
                new_values[state] = max_value
            values = new_values

        # recall that environment dynamics are available as full tensors:
        # w. `self.env.get_dynamics_tensor()`, or via `get_transition(s, a, s')`

        # Make sure you return an object extending the Policy interface, so
        # that you are able to render the policy and we can evaluate it.

        policy_decision = [0 for _ in range(nS)]
        for state in range(nS):
            max_value = -np.inf
            best_choice = None
            for action in range(nA):
                value = 0
                for next_state in range(nS):
                    next_state_probability = self.env.get_transition(state, action, next_state)
                    reward = self.env.get_reward(state, action, next_state)
                    value += (next_state_probability * (reward + gamma * values[next_state]))
                if value > max_value:
                    max_value = value
                    best_choice = action
            policy_decision[state] = best_choice

        policy = ValuePolicy(values, policy_decision)
        return policy


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
        self.q_table = np.zeros((self.env.num_states(), self.env.num_actions()))
        self.visit_counts = np.zeros((self.env.num_states(), self.env.num_actions()))
        self.average_rewards = 0
        self.initial_state_values = np.array([])

    def _epsilon_greedy_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.choice(self.env.num_actions())  # Exploration
        return np.argmax(self.q_table[state])  # Exploitation

    def train(self, gamma, steps, eps, explore_starts=False, **kwargs) -> EpsGreedyPolicy:
        step = 0
        episode_num = 0

        state, info = self.env.reset(randomize=explore_starts)
        episode_rewards = []
        episode = []

        episodes = 0
        total_reward = 0
        v_pi_t_s0 = 0
        rewards_per_episode = []

        while step < steps:
            action = self._epsilon_greedy_action(state, eps)
            succ, rew, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            episode.append((state, action, rew))
            total_reward += rew
            state = succ
            step += 1

            if logger is not None:
                logger.write({"rew": rew, "termination": terminated}, step)

            if done:
                # Calculate returns and update Q-values for every visit
                G = 0
                for t in reversed(range(len(episode))):
                    state, action, reward = episode[t]
                    G = reward + gamma * G  # Calculate return from this step onwards

                    # Every-Visit MC updates (average the returns)
                    self.visit_counts[state, action] += 1
                    self.q_table[state, action] += (G - self.q_table[state, action]) / self.visit_counts[state, action]

                # v^{pi_t}(s_0) with Exponential Moving Average (EMA)
                # https://groww.in/p/exponential-moving-average
                discounted_return = total_reward * gamma
                # v_pi_t_s0 = (1 - lr) * v_pi_t_s0 + lr * discounted_return
                # self.initial_state_values = np.append(self.initial_state_values, v_pi_t_s0)

                episode_rewards.append(G)
                episode = []
                episode_num += 1
                state, info = self.env.reset(randomize=explore_starts)

        self.average_rewards = np.cumsum(self.env.episode_rewards) / (np.arange(len(self.env.episode_rewards)) + 1)
        return EpsGreedyPolicy(self.q_table, eps)

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
    LimitedEnv = EnvWrapper(gym.make('CliffWalking-v0'), max_samples=10)

    """
        Logging example - walk through the CliffWalking environment
        randomly and log reward collected each step to the directory below.

        The logs should be located at results/test/logs.csv. 
    """
    log_dir = "results/test/"
    logger = Logger(log_dir)
    # VITrainer(CliffWalking).train(gamma=1.0, steps=42)
    # df = pd.read_csv(log_dir + "logs.csv", sep=";")
    # print(df.head(10))
    # QLTrainer(CliffWalking).train(gamma=1.0, steps=42, eps=0.42, lr=0.42, logger=logger)
    # df = pd.read_csv(log_dir + "logs.csv", sep=";")
    # print(df.head(10))

    """ 
        You can also use the `render_mode="human"` argument for Gymnasium to
        see an animation of your agent's decisions.
    """
    #AnimatedEnv = EnvWrapper(gym.make('FrozenLake-v1', map_name='4x4'
    #                                  , render_mode='human'),
    #                         max_samples=-1)
    # AnimatedEnv.reset()
    # Walk around randomly for a bit
    moves = [0, 1, 2, 3]
    # for i in moves:
    #    obs, rew, done, trunc, _ = AnimatedEnv.step(i)
    #    print(rew)
    #    if done:
    #        AnimatedEnv.reset()

    """
        Rendering example - using env.render_policy() to get a value heatmap as
        well as the greedy actions w.r.t. the policy values.
    """


    def render_random(env):
        """
            Plots heatmap of the state values and arrows corresponding to actions on `env`
        """
        env.reset(randomize=False)
        mc_trainer = MCTrainer(env)
        policy2 = mc_trainer.train(gamma=0.99, steps=100000, eps=0.1, explore_starts=False)
        env.render_policy(policy2, label="monte_carlo")

    render_random(FrozenLake)
