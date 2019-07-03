
import pygame

playerRad = 15
ballRad = 10


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
            if key == 'RIGHT':
                return self.pressed_keys[275]
            if key == 'DOWN':
                return self.pressed_keys[274]
            if key == 'LEFT':
                return self.pressed_keys[276]
            if key == 'RCTRL':
                return self.pressed_keys[305]
            return self.pressed_keys[ord(key)]
        else:
            return self.pressed_keys[key]

    def shutdown(self):
        pygame.quit()
