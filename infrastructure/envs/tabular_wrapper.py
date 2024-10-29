from datetime import datetime
import gymnasium as gym
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


from .visualization import render_heatmap, render_actions
from ..utils.logger import Logger

class EnvWrapper:
    """
        Environment wrapper for CliffWalk and FrozenLake environments that 
        provides the following functionality:

        1) Access to environment dynamics and other miscellaneous environment 
        information, number of states, actions. See methods:
            - num_states()
            - num_actions()
            - get_dynamics_tensor()
            - get_reward_tensor()
            - get_transition()
            - get_reward()

        
        2) Episodic interaction through methods `step()` / `reset()`, support for exploring
        starts randomly with `reset(randomize=True)`.

        3) Support for rendering Policy objects via `render_policy()`
        (currently works only for FrozenLake and CliffWalking).
    """

    def __init__(self, env, max_samples=-1, logging=False,
            log_steps=500, log_dir="results/"):

        """
            `max_samples` specify the number of `step()` calls you are able to
            make in this environment. The default value of -1 signals unlimited
            sampling from this environment.

            The remaining parameters control the frequency of logging as well
            as the path and name of the resulting csv data.
        """

        self.env = env 

        # name of the env instance
        self.name = self.env.unwrapped.spec.id

        self.logging = logging
        self.log_steps = log_steps

        if log_dir[-1] != "/":
            log_dir += "/"

        if logging:
            self.logger = Logger(log_dir)

        self.steps = 0
        self.ep_step = 0
        self.episodes = 0
        self.max_steps = max_samples

        self.episode_rewards = []
        self.episode_lengths = []

        self._initialize_dynamics()



    """
        ENV INTERFACE
    """


    def num_states(self):
        """
            Returns the number of states in the environment.
            The states are always indices from 0 to num_states() - 1.
        """
        return self.env.observation_space.n

    def num_actions(self):
        """
            Returns the number of actions in the environment.
        """
        return self.env.action_space.n

    def get_dynamics_tensor(self):
        """
            Returns Numpy array P of shape (S,A,S),
            where an entry P[s,a,s'] corresponds to p(s'|s,a)
        """
        return self.dynamics_tensor


    def get_reward_tensor(self):
        """
            Returns Numpy array R of shape (S,A,S),
            where an entry R[s,a,s'] corresponds to r(s,a,s')
        """
        return self.reward_tensor

    def get_transition(self, state, act, succ):
        """
            Returns p(succ | state, act)
        """
        return self.dynamics_tensor[state, act, succ]

    def get_reward(self, state, act, succ):
        """
            Returns r(state, act, succ)
        """
        return self.reward_tensor[state, act, succ]

    def render(self):
        """
            Renders the environment.
        """
        return self.env.render()

    def step(self, action):
        """
            Play `action` in the environment and return the tuple 
            (successor state, reward, terminated, truncated, info).
            See Gymnasium API documentation for more information.

            Additionally tracks and logs rewards for each episode.
        """

        if self.max_steps != -1 and self.steps >= self.max_steps:
            raise RuntimeError(f"Env {self.name} has been sampled over"
                               f" max_steps = {self.max_steps} times.")


        res = self.env.step(action)
        rew = res[1]

        self.episode_rewards[-1] += rew
        self.episode_lengths[-1] += 1

        self.steps += 1
        self.ep_step += 1

        if self.logging and self.steps % self.log_steps == 0: 
            self.log_trajectories()

        return res
    
    def reset(self, seed=None, randomize=False):
        """
            Resets the environment, potentially seeding it.
            The flas `randomize` specifies that the new starting state should be
            randomized. All non-terminal states are equally likely to be chosen.
        """
        self.episode_rewards.append(0)
        self.episode_lengths.append(0)

        self.episodes += 1
        self.ep_step = 0


        if randomize:
            nS = self.num_states()
            self.env.unwrapped.initial_state_distrib = np.zeros(nS)
            for state in self.nonterminal:
               self.env.unwrapped.initial_state_distrib[state] = 1/len(self.nonterminal)

        # if not randomizing fix init distribution
        else:
            self.env.unwrapped.initial_state_distrib = self.init_distr

        return self.env.reset(seed=seed)


    def render_policy(self, policy, label=""):
        """
            Accepts a policy object that implements `play()` and `raw()` methods.

            Renders arrows representing the played actions on the environment
            along with a heatmap of state values from `raw()` in the same figure.
        """
        if self.episodes == 0:
            raise RuntimeError("Please call reset on the environment before"
                               " rendering the policy.")

        # Switch env to rgb_array mode to render one frame
        prev_mode = self.env.unwrapped.render_mode 
        self.env.unwrapped.render_mode = "rgb_array"
        frame = self.env.render()
        self.env.unwrapped.render_mode = prev_mode

        # Get values of states
        nS = self.num_states()
        state_values = [ policy.raw(obs) for obs in range(nS) ]

        r, c = self._map_dimensions()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # Translate action indices to movement directions (the envs differ in the mapping)
        direction_map = self._get_direction_mapping()
        action_directions = [direction_map[policy.play(obs, greedy=True)] for obs in range(nS) ]

        render_actions(frame, action_directions, r, c, self.get_name(), label, ax1)
        render_heatmap(frame, state_values, r, c, self.get_name(), label, ax2)

        plt.tight_layout()
        plt.show()


    """
        HELPER METHODS 
    """

    def clear_stats(self):
        self.steps = 0
        self.ep_step = 0
        self.episodes = 0

        self.episode_rewards = []
        self.episode_lengths = []

    def get_name(self):
        return self.name

    def log_trajectories(self):
        
        # Log only finished episodes
        i = len(self.episode_lengths)

        if self.ep_step != 0:
            i -= 1

        if i == 0: 
            print("No finished episodes in the last logging period.")
            return

        lengths = self.episode_lengths[:i]
        total_rewards = self.episode_rewards[:i]

        mean_length = np.mean(lengths)
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)

        data = {
                "Episode Length" : mean_length,
                "Average Reward" : mean_reward,
                "Std Reward" : std_reward
                }
        
        print(self.name, "- ", end="")
        self.logger.write(data, self.steps)

        self.episode_lengths = self.episode_lengths[i:]
        self.episode_rewards = self.episode_rewards[i:]


    # initialize dynamics and reward tensors from gym data
    def _initialize_dynamics(self):
        if not hasattr(self.env.unwrapped, 'P'):
            raise AttributeError("Gym environment passed to the EnvWrapper",
                                 "does not provide dynamics.")

        nS, nA = self.num_states(), self.num_actions()

        reward_tensor = np.zeros((nS, nA, nS))
        dynamics_tensor = np.zeros((nS, nA, nS))

        # certain states should be absorbing, but the dynamics in P do not match
        # fix this manually here
        absorbing_states = [47] if "CliffWalking" in self.name else []


        # this will store all non-terminal states, so we can explore starts
        # from them later.
        nonterminal_states = []
        gym_matrix = self.env.unwrapped.P

        for s in range(nS):
            terminal = True 
            for a in range(nA):
                if s in absorbing_states:
                    dynamics_tensor[s,a,s] = 1.0

                else:
                    for p, succ, r, _ in gym_matrix[s][a]:
                        dynamics_tensor[s,a,succ] += p
                        reward_tensor[s,a,succ] += r

                        if succ != s:
                            terminal = False

            if not terminal:
                nonterminal_states.append(s)

        self.dynamics_tensor = dynamics_tensor
        self.reward_tensor = reward_tensor
        self.nonterminal = nonterminal_states
        self.init_distr = self.env.unwrapped.initial_state_distrib

    # get number of rows/columns in the rendered map
    def _map_dimensions(self):
        height, width = 0, 0
        if "FrozenLake" in self.name:
            nS = self.num_states()
            height = int(sqrt(nS))
            width = height
        elif 'CliffWalking' in self.name:
            height = 4
            width = 12

        return height, width


    # translate action index to the actual movement direction
    def _get_direction_mapping(self):
        action_dict = {}
        if "FrozenLake" in self.name:
            action_dict = {
                0 : "LEFT",
                1 : "DOWN",
                2 : "RIGHT",
                3 : "UP"
            }

        elif 'CliffWalking' in self.name:
            action_dict = {
                0 : "UP",
                1 : "RIGHT",
                2 : "DOWN",
                3 : "LEFT"
            }
        return action_dict
            
