#!/bin/bash
python FlappyBirdDQN.py --model dqn > dqn.log &
python FlappyBirdDQN.py --model dqnnature > dqnnature.log &
python FlappyBirdDQN.py --model ddqn > ddqn.log &
python FlappyBirdDQN.py --model duelingdqn > duelingdqn.log &
python FlappyBirdDQN.py --model prioritydqn > prioritydqn.log &
python FlappyBirdDQN.py --model actorcritic > actorcritic.log &
