import pygame
import sys

# Images: {
#   numbers: (img'0',img'1',img'2',img'3',img'4',img'5',img'6',img'7',img'8',img'9'),
#   base: img'base'
#   background: img'background'
#   player: (img'up', img'mid', img'down')
#   pipe: (img'up_pipe', img'down_pipe')
#}
# Sounds: {die, hit, point, swoosh, wing}
# Hitmasks: {
#   pipe:(up, down) 是一个矩阵，元素为True, False，矩阵大小等于对应图片的像素点
#   player:(up, mid, down)
# }
def load():
    # path of player with different states
    # three different pictures of bird's different pose.
    PLAYER_PATH = (
            'assets/sprites/redbird-upflap.png',
            'assets/sprites/redbird-midflap.png',
            'assets/sprites/redbird-downflap.png'
    )

    # path of background
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # path of pipe
    PIPE_PATH = 'assets/sprites/pipe-green.png'

    # path of base
    BASE_PATH = 'assets/sprites/base.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    # numbers sprites for score display
    # # pygame.image.load:
    # Load an image from a file source. You can pass either a filename or a Python file-like object.
    # # convert_alpha()
    # change the pixel format of an image including per pixel alphas
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load(BASE_PATH).convert_alpha()

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = (
        pygame.image.load(PLAYER_PATH[0]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[1]).convert_alpha(),
        pygame.image.load(PLAYER_PATH[2]).convert_alpha(),
    )

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # sounds
    # # sys.platform {linux, win32, win64, ...}
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    #SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    #SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    #SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    #SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    #SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    return IMAGES, SOUNDS, HITMASKS


def getHitmask(image):
    """
    returns a hitmask using an image's alpha.
    if image is <Surface(3*2*4)>, than return could be
    [[True, False],
     [True, False],
     [False, False]]
    """
    # if image is <Surface(52*320*32 SW)>, than image.get_width() is 52,
    # image.get_height() is 320
    mask = []
    for x in range(image.get_width()):  # for image's every row, add a []
        mask.append([])
        # for one row's every pixel, add a True(non-transparent)/False(transparent)
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    # # get_at()
    # get the color value at a single pixel
    # get_at((x, y)) -> Color
    # return Color -> (red, green, blue, alpha)
    # alpha value means transparency and 255 means non-transparent, 0 means transparent.
    return mask
