import pygame
from math import sin, cos, pi

class DebugSurf():
    def __init__(self):
        self.surf = pygame.Surface((256,256 + 16))

    def drawMove(self, probs, selected, team, win_prob = None):
        # Gives a surface to show the prob of each move inputted.
        self.surf.fill((0, 0, 0))
        for i in range(9):
            if team == "red":
                colour = (255, 0, 0) if i == selected else (255, 128, 128)
            elif team == "blue":
                colour = (0, 0, 255) if i == selected else (128, 128, 255)
            else:
                raise ValueError
            if i == 0:
                pygame.draw.circle(self.surf, colour, (128, 128), int(32 * probs[0]))
            else:
                x = 2 * pi * (i - 1) / 8
                start = (128 + 48 * sin(x), 128 + 48 * cos(x))
                vect  = ((1 + 64 * probs[i]) * sin(x), (1 + 64 * probs[i]) * cos(x))
                end   = (start[0] + vect[0], start[1] + vect[1])
                pygame.draw.line(self.surf, colour, start, end, 32)
        if win_prob != None:
            if team == "red":
                colour = (255, 0, 0)
            elif team == "blue":
                colour = (0, 0, 255)
            else:
                raise ValueError
            pygame.draw.rect(self.surf, colour, pygame.Rect(32, 256, (256 - 64) * win_prob, 16))
            # Draw halfway line in dark colour 
            if win_prob > 0.5:
                pygame.draw.line(self.surf, colour, (128, 256 - 16), (128, 256), 3 )
            if team == "red":
                colour = (255, 128, 128)
            elif team == "blue":
                colour = (128, 128, 255)
            else:
                raise ValueError
            pygame.draw.rect(self.surf, colour, pygame.Rect((256 - 64) * win_prob + 32, 256, (256 - 64) * (1 - win_prob), 16))
            # Draws a halfway line of lighter colour if winprob is low
            if win_prob <= 0.5:
                pygame.draw.line(self.surf, colour, (128, 256 - 16), (128, 256), 3 )
