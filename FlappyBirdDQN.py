# -----------------------------------------#
# Project: Deep Q-Learning on Flappy Bird  #
# Author: Chaney                           #
# Date: 2019.3.9                           #
# -----------------------------------------#

import sys
sys.path.append("game/")
import cv2
import game.wrapped_flappy_bird as game

from BrainDQNNature import BrainDQNNature
# from BrainDQNNature_CC import BrainDQN
# from BrainPrioritizedReplyDQN import BrainPrioritizedReplyDQN
from BrainDQN import BrainDQN
from BrainDoubleDQN import BrainDoubleDQN
from BrainPolicyGradient import BrainPolicyGradient
# import BrainDoubleDQN
import BrainDuelingDQN
import numpy as np

def preprocess(observ):
    observ = cv2.cvtColor(cv2.resize(observ, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observ = cv2.threshold(observ, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observ, (80, 80, 1))

def playFlappyBird():
    # Step 1: init BrainDQN
    actionNum = 2
    gameName = 'bird'
    brain = BrainDQNNature(actionNum, gameName)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1, 0])  # do nothing
    observation0, reward0, terminal, curScore = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while True:
        action = brain.getAction()
        nextObserv, reward, terminal, curScore = flappyBird.frame_step(action)
        nextObserv = preprocess(nextObserv)
        brain.setPerception(nextObserv, action, reward, terminal, curScore)

def main():
    playFlappyBird()

if __name__ == '__main__':
    main()