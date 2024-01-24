from flask import Flask, render_template, request
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('SVG')
import random
import imageio
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class GridEnvironment(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'video.frames_per_second': 10}

    def __init__(self, render_mode='rgb_array'):

        self.render_mode = render_mode
        self.canvas = FigureCanvas(plt.gcf())
        self.observation_space = spaces.Discrete(36)
        self.action_space = spaces.Discrete(4)
        self.max_timesteps = 100
        self.reward = 0
        self.agent_pos = [0, 0]
        self.goal_pos = [5, 5]
        self.gamma = 0.8
        self.environment_height = 6
        self.environment_width = 6

        reward1 = [0, 3]
        reward2 = [2, 1]
        reward3 = [1, 2]
        reward4 = [2, 3]
        reward5 = [5, 0]
        reward6 = [5, 5]
        reward7 = [4, 4]
        reward8 = [4, 2]
        reward9 = [2, 5]
        reward10 = [1, 0]
        reward11 = [3, 1]
        reward12 = [0, 5]
        reward13 = [5, 2]
        reward14 = [5, 3]
        reward15 = [5, 4]
        reward16 = [3, 3]

        self.reward_grid = np.zeros((6, 6))
        self.reward_grid[tuple(reward1)] = 1
        self.reward_grid[tuple(reward2)] = 3
        self.reward_grid[tuple(reward4)] = 1
        self.reward_grid[tuple(reward5)] = 1
        self.reward_grid[tuple(reward6)] = 50
        self.reward_grid[tuple(reward8)] = 3
        self.reward_grid[tuple(reward9)] = 2
        self.reward_grid[tuple(reward10)] = 2
        self.reward_grid[tuple(reward11)] = 3
        self.reward_grid[tuple(reward13)] = 2
        self.reward_grid[tuple(reward15)] = 4

        self.timestep = 0
        self.rewards = 0

        self.state = np.zeros((6, 6))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        observation = self.state.flatten()

    def reset(self, **kwargs):

        self.timestep = 0
        self.reward = 0
        self.agent_pos = [0, 0]
        self.goal_pos = [5, 5]

        self.state = np.zeros((6, 6))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        observation = self.state.flatten()

        reward1 = [0, 3]
        reward2 = [2, 1]
        reward3 = [1, 2]
        reward4 = [2, 3]
        reward5 = [5, 0]
        reward6 = [5, 5]
        reward7 = [4, 4]
        reward8 = [4, 2]
        reward9 = [2, 5]
        reward10 = [1, 0]
        reward11 = [3, 1]
        reward12 = [0, 5]
        reward13 = [5, 2]
        reward14 = [5, 3]
        reward15 = [5, 4]
        reward16 = [3, 3]

        self.reward_grid = np.zeros((6, 6))
        self.reward_grid[tuple(reward1)] = 1
        self.reward_grid[tuple(reward2)] = 3
        self.reward_grid[tuple(reward4)] = 1
        self.reward_grid[tuple(reward5)] = 1
        self.reward_grid[tuple(reward6)] = 50
        self.reward_grid[tuple(reward8)] = 3
        self.reward_grid[tuple(reward9)] = 2
        self.reward_grid[tuple(reward10)] = 2
        self.reward_grid[tuple(reward11)] = 3
        self.reward_grid[tuple(reward13)] = 2
        self.reward_grid[tuple(reward15)] = 4

        info = {}

        return observation

    def step(self, action, type_env):

        Left = 0
        Right = 1
        Up = 2
        Down = 3
        done = False
        rand_act = action
        if type_env == 'deterministic':
            if action == Left:
                self.agent_pos[1] -= 1
            if action == Right:
                self.agent_pos[1] += 1
            if action == Up:
                self.agent_pos[0] -= 1
            if action == Down:
                self.agent_pos[0] += 1

        if type_env == 'stochastic':
            rand_num = random.uniform(0,1)
            if rand_num > 0.9:
                rand_act = np.random.choice(env.action_space.n)
            else:
                rand_act = action

            if rand_act == Left:
                self.agent_pos[1] -= 1
            if rand_act == Right:
                self.agent_pos[1] += 1
            if rand_act == Up:
                self.agent_pos[0] -= 1
            if rand_act == Down:
                self.agent_pos[0] += 1

        self.agent_pos = np.clip(self.agent_pos, 0, 5)
        self.state = np.zeros((6, 6))
        self.state[tuple(self.agent_pos)] = 1
        self.state[tuple(self.goal_pos)] = 0.5
        observation = self.state.flatten()

        available_reward = self.reward_grid[tuple(self.agent_pos)]
        self.reward_grid[tuple(self.agent_pos)] = 0
        reward = available_reward
        self.timestep += 1

        terminated = True if np.all(
            (self.timestep >= self.max_timesteps) or (self.agent_pos == self.goal_pos)) else False
        truncated = True if np.all((self.agent_pos >= 5) & (self.agent_pos <= 0)) else False

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self, mode='rgb_array', close=False):
        plt.imshow(self.state)

        self.canvas.draw()
        pil_image = Image.frombytes('RGB', self.canvas.get_width_height(), self.canvas.tostring_rgb())

        if mode == 'human':
            plt.imshow(self.state)
            plt.axis('off')
        elif mode == 'rgb_array':
            return np.asarray(pil_image)

    def plots(self, rewards, rewards_evaluate):
        plt.figure(figsize=(10, 8))
        plt.plot(rewards_evaluate, linewidth=2, color='red')
        plt.xlabel('Episodes', fontsize=28)
        plt.ylabel('Rewards per episode', fontsize=28)
        plt.title('Rewards Per Episode', fontsize=28)
        plt.grid()
        plt.savefig('./static/images/plot_eval.png')

        plt.figure(figsize=(10, 8))
        plt.plot(rewards, linewidth=2, color='red')
        plt.xlabel('Episodes', fontsize=28)
        plt.ylabel('Rewards per episode', fontsize=28)
        plt.title('Rewards Per Episode', fontsize=28)
        plt.grid()
        plt.savefig('./static/images/plot_train.png')
        # plt.show()


class dqnAgent():

    def __init__(self):
        pass

    def update_q_qval(self, state, action, next_action, reward, next_state, gamma, alpha):
        q = alpha * (reward + (gamma * np.max(q_table[next_state, :])) - q_table[(state, action)])
        q_table[(state, action)] += q

    def update_sarsa_qval(self, state, action, next_action, reward, next_state, gamma, alpha):
        q = alpha * (reward + (gamma * q_table[next_state, next_action]) - q_table[(state, action)])
        q_table[(state, action)] += q

    def update_dq_qval(self, state, action, next_action, reward, next_state, gamma, alpha):
        if np.random.rand() > 0.5:
            act = np.argmax(q_table[next_state, :])
            max_q_value = q_tableb[next_state][act]

            q_table[(state, action)] = q_table[(state, action)] + alpha * (
                        reward + gamma * max_q_value - q_table[(state, action)])

        else:
            act = np.argmax(q_tableb[next_state, :])
            max_q_value = q_table[next_state][act]

            q_tableb[(state, action)] = q_tableb[(state, action)] + alpha * (
                    reward + gamma * max_q_value - q_tableb[(state, action)])

    def act(self, epsilon, state, algo):
        rand_num = random.uniform(0, 1)
        if rand_num < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            table = q_table[state, :]
            if algo == "Double Q Learning":
                table = q_table[state]+q_tableb[state]
            action = np.argmax(table)
        return action

    def evaluate(self, algo, timesteps = 100):
        eval_rewards = []
        frames_eval = []
        total_steps_eval = []

        for i in range(10):
            print('-------', i)
            state = env.reset()
            state = np.argmax(state)
            rew_eval = 0
            terminated = False
            action = agent.act(0.001, state, algo)
            if i == 1:
                frames_eval.append(env.render())

            for t in range(timesteps):

                observation, reward, terminated, truncated, info = env.step(action, 'deterministic')
                next_state = np.argmax(observation)
                next_action = agent.act(0.001, next_state, algo)

                state = next_state
                action = next_action

                rew_eval += reward
                if i == 1:
                    frames_eval.append(env.render())

                if truncated or terminated:
                    break

            eval_rewards.append(rew_eval)
            total_steps_eval.append(t)

        imageio.mimsave('./static/videos/eval_render.mp4', frames_eval, fps=5)
        print("Eval Saved video.")
        return eval_rewards


env = GridEnvironment(render_mode='rgb_array')
agent = dqnAgent()
q_table = np.zeros((env.observation_space.n, env.action_space.n))
q_tableb = np.zeros((env.observation_space.n, env.action_space.n))


def run_gridworld(selected_env, episodes, alpha=0.15, gamma=0.9, algo="Q Learning"):

    algo = algo
    alpha = float(alpha)
    gamma = float(gamma)
    epsilon = 1.0
    min_epsilon = 0.01
    episode = int(episodes)
    timesteps = 100
    rewards = []
    total_steps = []
    eps = []
    frames = []

    for i in range(episode):
        print('-------', i)
        state = env.reset()
        state = np.argmax(state)
        rew = 0
        terminated = False
        action = agent.act(epsilon, state, algo)

        if i in [1, 100, 500, episode-2]:
            frames.append(env.render())

        for t in range(timesteps):
            observation, reward, terminated, truncated, info = env.step(action,'deterministic')
            next_state = np.argmax(observation)
            next_action = agent.act(epsilon, next_state, algo)
            if algo == "Q Learning":
                agent.update_q_qval(state, action, next_action, reward, next_state, gamma, alpha)
            elif algo == "Double Q Learning":
                agent.update_dq_qval(state, action, next_action, reward, next_state, gamma, alpha)
            else:
                agent.update_sarsa_qval(state, action, next_action, reward, next_state, gamma, alpha)

            state = next_state
            action = next_action

            rew += reward

            if i in [1, 100, 500, 999]:
                frames.append(env.render())

            if truncated or terminated:
                break

            if i >= 950:
                print(action)

        rewards.append(rew)
        total_steps.append(t)
        epsilon = max(epsilon * 0.995, min_epsilon)
        eps.append(epsilon)

    imageio.mimsave('./static/videos/train_render.mp4',frames,fps=30)
    print("Saved video.")

    rewards_evaluate = agent.evaluate(algo)
    env.plots(rewards, rewards_evaluate)


app = Flask(__name__, template_folder='template')


@app.route('/')
def index():
    # Define a list of options for the dropdown menu
    options = ['Grid World']
    options_a = ['Q Learning', 'SARSA', 'Double Q Learning']

    # Render the HTML template and pass the options to it
    return render_template('index.html', options=options, options_a=options_a)

@app.route('/selection', methods=['POST'])
def selection():
    # Get the value of the selected option from the form
    selected_env = request.form['env']
    selected_algo = request.form['algo']
    episodes = request.form['episodes']
    learning_rate = request.form['learning_rate']
    gamma = request.form['gamma']
    run_gridworld(selected_env,episodes, learning_rate, gamma, selected_algo)
    print("execution completed")

    return render_template('selection.html', selected_option=selected_env, episodes=episodes, learning_rate=learning_rate, gamma=gamma, selected_algo=selected_algo)


if __name__ == '__main__':
    app.run(debug=True)
