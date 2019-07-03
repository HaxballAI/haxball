
import pygame
from pygame import gfxdraw
from game_simulator import gameparams as gp





class GameWindow:

    def __init__(self, winWidth, winHeigh, fps = 60):
        self.height = winHeigh
        self.width = winWidth

        self.clock = pygame.time.Clock()
        self.fps = fps

        self.rip = False

        pygame.init()

        self.win = pygame.display.set_mode( (self.width, self.height ) )
        pygame.display.set_caption( "TEST DISPLAY" )

    def drawThings(self, things):
        # Things is of a very specific format that should not be violated

        self.win.fill( (0, 0, 0 ) )

        # draws ball area
        pygame.draw.rect(self.win, gp.pitchcolour, (gp.pitchcornerx, gp.pitchcornery, gp.pitchwidth, gp.pitchheight))
        #draws area behind goal
        pygame.draw.rect(self.win, gp.pitchcolour, (gp.pitchcornerx - 30, gp.goalcornery, 30, gp.goalsize))
        pygame.draw.rect(self.win, gp.pitchcolour, (gp.windowwidth - gp.pitchcornerx, gp.goalcornery, 30, gp.goalsize))


        for redP in things[0]:
            intP = tuple( map( int, redP ) )
            pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1], gp.playerradius-gp.kickingcirclethickness, gp.redcolour)
            pygame.gfxdraw.aacircle(self.win, intP[0], intP[1], gp.playerradius-gp.kickingcirclethickness, gp.redcolour)

        for blueP in things[1]:
            intP = tuple( map( int, blueP ) )
            pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1], gp.playerradius-gp.kickingcirclethickness, gp.bluecolour)
            pygame.gfxdraw.aacircle(self.win, intP[0], intP[1], gp.playerradius-gp.kickingcirclethickness, gp.bluecolour)

        for ballP in things[2]:
            intP = tuple( map( int, ballP ) )
            pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1], gp.ballradius+2, (0, 0, 0))
            pygame.gfxdraw.aacircle(self.win, intP[0], intP[1], gp.ballradius+2, (0, 0, 0))
            pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1], gp.ballradius, (255, 255, 255))
            pygame.gfxdraw.aacircle(self.win, intP[0], intP[1], gp.ballradius, (255, 255, 255))



        self.clock.tick(self.fps)
        pygame.display.update()

    def getInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.rip = True

    def shutdown(self):
        pygame.quit()
