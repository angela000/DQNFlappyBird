from maze_dqn.maze import Maze
from maze_dqn.brain import DeepQNetwork
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import time

from maze_dqn.maze import MAZE_W
from maze_dqn.maze import MAZE_H

MAX_EPISODE = 400  # 400
REPLY_START_SIZE = 1000  # 1000
COUNTERS_SIZE = 10
UPDATE_FREQUENCY = 5


def run_maze():
    step = 0
    flag = True
    counters = []  # store the last ten episodes' count
    average_counts = []  # store all the average counts on counters

    for episode in range(MAX_EPISODE):
        print('episode:' + str(episode))
        # initial observation
        observation = env.reset_maze()
        # counter for one episode
        count = 0

        while True:
            # RL choose action based on observation
            action = RL.choose_action(phi(observation), step)

            # RL take action and get next observation and reward
            reward, done, observation_ = env.update(action)

            RL.store_transition(phi(observation), action, reward, phi(observation_))

            # start learning on condition 'step > REPLY_START_SIZE', so that we can have
            # enough transition stored for sampling.
            # 'step % UPDATE_FREQUENCY == 0' is for frame-skipping technique.
            if (step > REPLY_START_SIZE) and (step % UPDATE_FREQUENCY == 0):
                # if flag:
                #     print('Learning...')
                #     flag = False
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

            # counter for one episode.
            count += 1

        if reward == 1:
            print('episode:' + str(episode) + ' | ' + str(count))
        else:
            print('episode:' + str(episode) + ' | ' + 'failed!')
            count = 200  # if agent failed to get reward, then step_count = 25.

        counter_add(counters, count)  # append the last COUNTERS_SIZE count to counters
        # average the last COUNTERS_SIZE counts and store them in average_counts
        average_counts.append(np.mean(counters))

    evaluation(average_counts)

    # end of game
    print('game over')


# # Evaluation and visualize the q value.
def evaluation(average_counts):
    actions_ = []
    q_evals_ = []
    observation = [np.array([[6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 6, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 6, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 6, 0, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 6, 0, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 6, 0, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 6, 0, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 6, -1, 0, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 6, 0, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 6, 0, 1]]),
                   np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 6, 1]])]
    for ob in observation:
        q_eval, action = RL.choose_action_eval(ob)
        actions_.append(action)
        q_evals_.append(q_eval)

    # store actions_ & average_counts in actions.txt
    fo = open("./logs/log_info.txt", "w")
    fo.write('average_countsstr:\n' + str(average_counts) + '\n\n' + 'actions_:\n' + str(actions_) +
             '\n\n' + 'q_evals_:\n' + str(q_evals_))
    fo.close()

    # visualize q value by arrow
    visualize_q_value(actions_, MAZE_W, MAZE_H, env.get_bad_point(), env.get_good_point())

    # plot the average_counts
    plt.plot(average_counts)
    plt.ylabel('steps to exit')
    plt.savefig("./logs/average_counts_dqn.png")


def counter_add(counters, count):
    if len(counters) >= COUNTERS_SIZE:
        counters.pop(0)  # pop(0) --> FIFO
    counters.append(count)


def visualize_q_value(actions, width, height, bad_point, good_point):
    for i in range(width):
        for j in range(height):
            if (i, j) not in bad_point and (i, j) not in good_point:
                try:
                    print(print_arrow(actions.pop()), end="")
                except:
                    pass
            else:
                print(" ", end="")
        print('\n')


# translate the action(number) to arrow to visualize the optimize Q value.
def print_arrow(action):
    arrow = '↑'
    if action == 1:
        arrow = '↓'
    if action == 2:
        arrow = '←'
    if action == 3:
        arrow = '→'

    return arrow


# function phi() : used to image preprocessing. Here it is a empty function.
def phi(observation):
    pass
    return observation


if __name__ == "__main__":
    # get the maze environment
    env = Maze()
    # get the DeepQNetwork Agent
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.01,
                      output_graph=True,
                      )
    # Calculate running time
    start_time = time.time()

    run_maze()

    end_time = time.time()
    running_time = (end_time - start_time) / 60

    fo = open("./logs/running_time_dqn.txt", "w")
    fo.write(str(running_time) + "minutes")
    fo.close()
