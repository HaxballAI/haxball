
import pygame
from pygame import gfxdraw
from game_simulator import gameparams as gp
from game_simulator import playeraction as pa





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
        # draws background
        pygame.draw.rect(self.win, gp.bordercolour, (0, 0, gp.windowwidth, gp.windowheight))
        # draws ball area
        pygame.draw.rect(self.win, gp.pitchcolour, (gp.pitchcornerx, gp.pitchcornery, gp.pitchwidth, gp.pitchheight))
        #draws area behind goal
        pygame.draw.rect(self.win, gp.pitchcolour, (gp.pitchcornerx - 30, gp.goalcornery, 30, gp.goalsize))
        pygame.draw.rect(self.win, gp.pitchcolour, (gp.windowwidth - gp.pitchcornerx, gp.goalcornery, 30, gp.goalsize))

        # draws goal lines
        pygame.draw.rect(self.win, gp.goallinecolour, (
        gp.pitchcornerx - gp.goallinethickness // 2, gp.pitchcornery - gp.goallinethickness // 2, gp.goallinethickness,
        gp.pitchheight + gp.goallinethickness))
        pygame.draw.rect(self.win, gp.goallinecolour, (
        gp.windowwidth - gp.pitchcornerx - gp.goallinethickness // 2, gp.pitchcornery - gp.goallinethickness // 2, gp.goallinethickness,
        gp.pitchheight + gp.goallinethickness))
        pygame.draw.rect(self.win, gp.goallinecolour, (
        gp.pitchcornerx - gp.goallinethickness // 2, gp.pitchcornery - gp.goallinethickness // 2, gp.pitchwidth + gp.goallinethickness,
        gp.goallinethickness))
        pygame.draw.rect(self.win, gp.goallinecolour, (
        gp.pitchcornerx - gp.goallinethickness // 2, gp.windowheight - gp.pitchcornery - gp.goallinethickness // 2,
        gp.pitchwidth + gp.goallinethickness, gp.goallinethickness))


        for redInfo in things[0]:
            intP = tuple( map( int, redInfo[0] ) )

            if redInfo[2]:
                pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, gp.kickingcirclecolour)
                pygame.gfxdraw.aacircle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, gp.kickingcirclecolour)

            else:
                pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, (0,0,0))
                pygame.gfxdraw.aacircle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, (0,0,0))


            pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1],
                gp.playerradius-gp.kickingcirclethickness, gp.redcolour)
            pygame.gfxdraw.aacircle(self.win, intP[0], intP[1],
                gp.playerradius-gp.kickingcirclethickness, gp.redcolour)



        for blueInfo in things[1]:
            intP = tuple( map( int, blueInfo[0] ) )
            if blueInfo[2]:
                pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, gp.kickingcirclecolour)
                pygame.gfxdraw.aacircle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, gp.kickingcirclecolour)

            else:
                pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, (0,0,0))
                pygame.gfxdraw.aacircle(self.win, intP[0], intP[1],
                    gp.kickingcircleradius, (0,0,0))


            pygame.gfxdraw.filled_circle(self.win, intP[0], intP[1],
                gp.playerradius-gp.kickingcirclethickness, gp.bluecolour)
            pygame.gfxdraw.aacircle(self.win, intP[0], intP[1],
                gp.playerradius-gp.kickingcirclethickness, gp.bluecolour)

        for ballInfo in things[2]:
            intP = tuple( map( int, ballInfo[0] ) )
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
