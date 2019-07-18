import pygame
from math import sin, cos, pi

def drawMove(probs, selected, team):
    # Gives a surface to show the prob of each move inputted.
    surf = pygame.Surface((256,256))
    surf.fill((0, 0, 0))
    for i in range(9):
        if team == "red":
            colour = (255, 0, 0) if i == selected else (255, 128, 128)
        elif team == "blue":
            colour = (0, 0, 255) if i == selected else (128, 128, 255)
        else:
            raise ValueError
        if i == 0:
            pygame.draw.circle(surf, colour, (128, 128), int(32 * probs[0]))
        else:
            x = 2 * pi * (i - 1) / 8
            start = (128 + 48 * sin(x), 128 + 48 * cos(x))
            vect  = ((1 + 64 * probs[i]) * sin(x), (1 + 64 * probs[i]) * cos(x))
            end   = (start[0] + vect[0], start[1] + vect[1])
            pygame.draw.line(surf, colour, start, end, 32)
    return surf

if __name__ == "__main__":
    pygame.init()
    m = drawMove([1.0] * 9,0)
    pygame.display.set_mode((256,256)).blit(m, (0, 0))
    pygame.display.update()
    input("Press enter:")
