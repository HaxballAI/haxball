import numpy as np

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], type(np.array([])) ):
        return flatten(S[0].tolist()) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def rotateVel(vel):
    return np.array( [ -vel[0], -vel[1] ] )

def rotatePos(pos):
    return np.array( [windowwidth - pos[0] , windowheight - pos[1] ] )

# Pygame stuff hasn't beed added properly yet!!!
windowwidth = 840
windowheight = 400
pitchwidth = 640
pitchheight = 260
goalsize = 110

# defines player numbers
# the first players are controlled manually
# this was added because it will end up being added anyways
# it also allows us to test the robustness of player-player collisions when there are large numbers of players
redteamsize = 1
blueteamsize = 1

# defines terminal game parameters
maxscore = 1

# game parameters for the player
playerradius = 15
playerbouncing = 0.5
playerinvmass = 0.5
playerdamping = 0.96
accel = 0.1
kickaccel = 0.07
kickstrength = 5

# game parameters for the ball
ballradius = 10
balldamping = 0.99
ballinvmass = 1
ballbouncing = 0.5

# parameters for the pitch drawing
redstart = (200, 200)
bluestart = (640, 200)
ballstart = (420, 200)
goalpostradius = 8
goalpostbouncingquotient = 0.5
goalpostborderthickness = 2
goallinethickness = 3
kickingcircleradius = 15
kickingcirclethickness = 2

# defines colors used in drawing the map
redcolour = (229, 110, 86)
bluecolour = (86, 137, 229)
ballcolour = (0, 0, 0)
goallinecolour = (199, 230, 189)
goalpostcolour = (150, 150, 150)
pitchcolour = (127, 162, 112)
bordercolour = (113, 140, 90)
kickingcirclecolour = (255, 255, 255)

# defines centre line properties
centrecircleradius = 70
centrecirclecolour = (199, 230, 189)
centrecirclethickness = 3
centrelinethickness = 3

# defines text properties
textcolour = (0, 0, 0)
textposition = (215, 25)

# defines relevant pitch coordinates for calculation
pitchcornerx = int(np.floor((windowwidth - pitchwidth) / 2))
pitchcornery = int(np.floor((windowheight - pitchheight) / 2))

goalcornery = int(np.floor((windowheight - goalsize) / 2))

redgoalpostup = (pitchcornerx, goalcornery)
redgoalpostdown = (pitchcornerx, goalcornery + goalsize)
bluegoalpostup = (windowwidth - pitchcornerx, goalcornery)
bluegoalpostdown = (windowwidth - pitchcornerx, goalcornery + goalsize)

goalposts = [redgoalpostup, redgoalpostdown, bluegoalpostup, bluegoalpostdown]

y1 = pitchcornerx - 30

z1 = pitchcornerx + pitchwidth
z2 = goalcornery

a1 = y1 + 2 * ballradius
a2 = int(np.floor(goalcornery - goallinethickness / 2))

b1 = z1
b2 = int(np.floor(goalcornery - goallinethickness / 2))

# defines the movespace of a player
movespacex = [playerradius, windowwidth - playerradius]
movespacey = [playerradius, windowheight - playerradius]

# defines the movespace of a ball
ballspacex = [pitchcornerx + ballradius, pitchcornerx + pitchwidth - ballradius]
ballspacey = [pitchcornery + ballradius, pitchcornery + pitchheight - ballradius]

# defines goal width
goaly = [goalcornery, goalcornery + goalsize]

#empirical stdev:
#
mean = np.array([388.827068, 5.59644414, 62.3458645, 0, 31.1729323, -3.78333705, 0.00379109019, 0.0158872452, -0.00758218039, 0, -0.00379109019, 0.0118169034])
stdev = np.array([176.56940949, 67.29775796, 92.34440333, 45.15357567, 108.40524482, 53.97645633, 1.57480229, 0.88057945, 0.95995111, 0.71889745, 1.63657848, 1.46776913])
