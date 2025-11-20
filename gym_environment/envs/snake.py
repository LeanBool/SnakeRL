import gymnasium as gym


def create_pygame_window():
    import sys, os

    sys.stdout = open(os.devnull,'w') # suppress pygame startup output
    sys.stderr = open(os.devnull,'w')

    import pygame

    pygame.init()
    window = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stdout__

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.VIDEORESIZE:
                window = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

class SnakeEnv(gym.Env):
    def __init__(self):
        create_pygame_window()