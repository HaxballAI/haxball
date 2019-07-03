
import pygame

playerRad = 15
ballRad = 10


class GameWindow:

    def __init__(self, winWidth, winHeigh, fps = 60):
        self.height = winHeigh
        self.width = winWidth

        self.clock = pygame.time.Clock()
        self.fps = fps

        pygame.init()

        self.win = pygame.display.set_mode( (self.width, self.height ) )
        pygame.display.set_caption( "TEST DISPLAY" )

    def drawThings(self, things):
        # Things is of a very specific format that should not be violated

        self.win.fill( (0, 0, 0 ) )

        for redP in things[0]:
            intP = tuple( map( int, redP ) )
            pygame.draw.circle(self.win, (255,0,0), intP, playerRad)

        for blueP in things[1]:
            intP = tuple( map( int, blueP ) )
            pygame.draw.circle(self.win, (0,0,255), intP, playerRad)

        for ballP in things[2]:
            intP = tuple( map( int, ballP ) )
            pygame.draw.circle(self.win, (255,255,255), intP, ballRad)


        self.clock.tick(self.fps)
        pygame.display.update()

    def shutdown(self):
        pygame.quit()
