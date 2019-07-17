import pygame
from pygame import gfxdraw
from game_simulator import gameparams as gp
from game_simulator import playeraction as pa

class GameWindow:
    def __init__(self, winWidth, winHeight, fps = 60):
        self.height = winHeight
        self.width = winWidth

        self.clock = pygame.time.Clock()
        self.fps = fps

        self.rip = False

        pygame.init()

        # Keys that are currently pressed down
        self.pressed_keys = pygame.key.get_pressed()

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

        # draws pitch borders
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

        cnt = 0
        # draws goalposts
        for goalpost in gp.goalposts:
            cnt += 1
            pygame.gfxdraw.filled_circle(self.win, goalpost[0], goalpost[1], gp.goalpostradius, (0, 0, 0))
            pygame.gfxdraw.aacircle(self.win, goalpost[0], goalpost[1], gp.goalpostradius, (0, 0, 0))
            goalpostcol = (0, 0, 0)
            if cnt <= 2:
                goalpostcol = (200, 150, 150)
            else:
                goalpostcol = (150, 150, 200)
            pygame.gfxdraw.filled_circle(self.win, goalpost[0], goalpost[1], gp.goalpostradius-gp.goalpostborderthickness, goalpostcol)
            pygame.gfxdraw.aacircle(self.win, goalpost[0], goalpost[1], gp.goalpostradius-gp.goalpostborderthickness, goalpostcol)

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

    def getInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.rip = True

    def updateKeys(self):
        # Updates the keys that are being pressed down
        self.pressed_keys = pygame.key.get_pressed()

    def isKeyPressed(self, key, is_int = False):
        # Checks whether a key is being pressed, input is a char or an int.
        # self.keys needs to be updated beforehand.
        if is_int == False:
            if key == 'UP':
                return self.pressed_keys[273]
            elif key == 'RIGHT':
                return self.pressed_keys[275]
            elif key == 'DOWN':
                return self.pressed_keys[274]
            elif key == 'LEFT':
                return self.pressed_keys[276]
            elif key == 'LALT':
                return self.pressed_keys[308]
            elif key == 'RALT':
                return self.pressed_keys[307]
            elif key == 'LCTRL':
                return self.pressed_keys[306]
            elif key == 'RCTRL':
                return self.pressed_keys[305]
            elif key == 'LSHIFT':
                return self.pressed_keys[304]
            elif key == 'RSHIFT':
                return self.pressed_keys[303]
            else:
                return self.pressed_keys[ord(key)]
        else:
            return self.pressed_keys[key]

    def shutdown(self):
        pygame.quit()
