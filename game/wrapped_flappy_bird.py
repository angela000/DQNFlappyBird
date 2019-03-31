import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# frames per secoond
FPS = 30
# background-black.png 288*512
SCREENWIDTH = 288
SCREENHEIGHT = 512

COUNTERS_SIZE = 10  # the number of episodes to average for evaluation. 10
AVERAGE_SIZE = 500  # the length of average_score to print a png. 500

# # pygame.init()
# initialize all imported pygame modules.

# # pygame.time.Clock
# create an object to help track time
# Clock() -> Clock

# # pygame.display.set_mode()
# Initialize a window or screen for display
# set_mode(resolution=(0,0), flags=0, depth=0) -> Surface

# # pygame.display.set_caption()
# Set the current window caption
# set_caption(title, icontitle=None) -> None

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79  # 404.48

PLAYER_WIDTH = IMAGES['player'][0].get_width()  # 34
PLAYER_HEIGHT = IMAGES['player'][0].get_height()  # 24
PIPE_WIDTH = IMAGES['pipe'][0].get_width()  # 52
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()  # 320
BACKGROUND_WIDTH = IMAGES['background'].get_width()  # 288

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])
# >> for item in cycle([0, 1, 2, 1]):
# >>   print item
# 0 1 2 1 0 1 2 1 0 1 2 1 0 1 2 1 ...


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH # 336 - 288 = 48

        # init的时候提前init好了两组pipe
        newPipes1 = getRandomPipe()  # e.g. [{'y':-190, 'x':298}, {'y':230, 'x':298}]
        newPipes2 = getRandomPipe()  # e.g. [{'y':-170, 'x':298}, {'y':250, 'x':298}]
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipes1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipes2[0]['y']},
        ]  # e.g. [{'y':-190, 'x':298}, {'y':-170, 'x':432}]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipes1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipes2[1]['y']},
        ]  # e.g. [{'y':230, 'x':298}, {'y':250, 'x':432}]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -4
        self.playerVelY = 0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

    def frame_step(self, input_actions):
        # # pygame.event.pump()
        # internally process pygame event handlers
        # pump() -> None
        # For each frame of your game, you will need to make some sort of call to the event queue.
        # This ensures your program can internally interact with the rest of the operating system.
        pygame.event.pump()

        reward = 0.1
        terminal = False

        # input_actions -> one-hot vector
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # player's movement
        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                # SOUNDS['wing'].play()
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        # 以上计算好应有的y方向上的速度，下面得出下一帧bird新的位置
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                # SOUNDS['point'].play()
                reward = 3

        # check if crash here
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                              'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)

        score_return = self.score

        if isCrash:
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -3

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        # print(self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2))
        # 注意这个image_data传的是新的

        return image_data, reward, terminal, score_return

        # # pygame.surfarray.array3d()
        # Copy pixels into a 3d array
        # array3d(Surface) -> array

        # # pygame.display.update()
        # Update portions of the screen for software displays
        # update(rectangle=None) -> None
        # update(rectangle_list) -> None
        # This function is like an optimized version of pygame.display.flip() for software displays.
        # It allows only a portion of the screen to updated, instead of the entire area.
        # If no argument is passed it updates the entire Surface area like pygame.display.flip().

        # # tick()
        # update the clock
        # tick(framerate=0) -> milliseconds
        # This method should be called once per frame. It will compute how many milliseconds have
        # passed since the previous call.
        # If you pass the optional framerate argument the function will delay to keep the game running
        # slower than the given ticks per second. This can be used to help limit the runtime speed of
        # a game. By calling Clock.tick(40) once per frame, the program will never run at more than 40
        # frames per second.


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs) - 1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)  # += 80.896
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    # # blit()
    # draw one image onto another
    # blit(source, dest, area=None, special_flags = 0) -> Rect
    # The coordinate origin of Pygame (0,0) is located in the upper left corner, and the X-axis is
    # from left to right, and the Y-axis is from the top down, and the unit is pixel.
    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()  # 32
    player['h'] = IMAGES['player'][0].get_height() # 32

    # if player crashes into ground
    # The coordinate origin of Pygame (0,0) is located in the upper left corner.
    if player['y'] + player['h'] >= BASEY - 1: # 撞地面
        return True
    else:   # 撞管道
        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

# 其实第一行代码应该就已经能比较粗略的进行碰撞比较了，利用pygame.Rect.clip()
def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)
    # # Rect.clip()
    # crops a rectangle inside another
    # clip(Rect) -> Rect
    # Returns a new rectangle that is cropped to be completely inside the argument Rect. If the two
    # rectangles do not overlap to begin with, a Rect with 0 size is returned.

    # 如果没有交集，return false，如果有交集，也不一定撞上（透明区域不算撞）
    if rect.width == 0 or rect.height == 0:
        return False

    # 计算交集区域在分别两个物体内部的位置
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    # 遍历交集区域的点，如果在分别两个物体内部的这个位置都是不透明的，就return True
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
    # usage in testing.
