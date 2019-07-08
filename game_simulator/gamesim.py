from game_simulator import entities
from game_simulator import gameparams
from game_simulator import playeraction

import numpy as np


class GameSim:
    def __init__(self, red_player_count, blue_player_count, ball_count, extraParams = {}, seed = -1):
        # Intialise the entities
        self.reds = [entities.Player("red", self.getRandomPositionInThePlayingField()) for i in range(red_player_count)]
        self.blues = [entities.Player("blue", self.getRandomPositionInThePlayingField()) for i in range(blue_player_count)]
        self.goalposts = [entities.GoalPost(np.array((gameparams.pitchcornerx, gameparams.goalcornery))),
                          entities.GoalPost(np.array((gameparams.pitchcornerx, gameparams.goalcornery + gameparams.goalsize))),
                          entities.GoalPost(np.array((gameparams.windowwidth - gameparams.pitchcornerx, gameparams.goalcornery))),
                          entities.GoalPost(np.array((gameparams.windowwidth - gameparams.pitchcornerx, gameparams.goalcornery + gameparams.goalsize)))]
        self.centre_block = entities.CentreCircleBlock(np.array(gameparams.ballstart))
        self.balls = [entities.Ball(self.getRandomPositionInThePlayingField()) for i in range(ball_count)]

        # Create useful groupings
        self.players = self.reds + self.blues
        self.moving_objects = self.players + self.balls

        # Game state info
        self.has_the_game_been_kicked_off = True

        self.red_last_goal = False
        # Flag showing whether a point was scored in the current step
        self.was_point_scored = False

        self.red_score = 0
        self.blue_score = 0

        # Number of elapsed frames
        self.frames = 0

        # Initialise the random seed iff seed != 1
        if seed != -1:
            np.random.seed(seed)

        # Sets extra information to do with. Probably a convention that I am
        # not following here.
        if "printDebug" in extraParams:
            self.printDebug = extraParams["printDebug"]
        else:
            self.printDebug = False

        if "printDebugFreq" in extraParams:
            self.printDebugFreq = extraParams["printDebugFreq"]
        else:
            self.printDebugFreq = 600

        if "auto score" in extraParams:
            self.auto_score = extraParams["auto score"]
        else:
            self.auto_score = False

        if "enforce kickoff" in extraParams:
            self.enforce_kickoff = extraParams["enforce kickoff"]
        else:
            self.enforce_kickoff = False

    def getRandomPositionInThePlayingField(self):
        return np.array([gameparams.pitchcornerx + (np.random.random_sample())*580, gameparams.pitchcornery + (np.random.random_sample())*200]).astype(float)

    def keepOutOfCentre(self, obj):
        # Moves an object out of the centre area. Called during kickoff
        vector = np.array([self.centre_block.pos[0] - obj.pos[0], self.centre_block.pos[1] - obj.pos[1]])
        distance = np.linalg.norm(vector)
        # I'm a bit confused as to what's happening here. First you move obj to not collide with
        # centreblock but then you also resolve the collision? Why? There is no collision happening...
        if distance <= self.centre_block.radius + obj.radius:
            obj.pos[0] = self.centre_block.pos[0] - vector[0] / np.linalg.norm(vector)
            obj.pos[1] = self.centre_block.pos[1] - vector[1] / np.linalg.norm(vector)
            self.resolveCollision(self.centre_block, obj, 1)
            self.centre_block.pos[0] = int(self.centre_block.pos[0]) # Idk why this even exists
            self.centre_block.pos[1] = int(self.centre_block.pos[1])

    def keepEntityInMovementSpace(self, obj, is_ball = 0):
        # should keep things on the board where the movement happens

        if is_ball == False:
            movement_space_x = [obj.radius, gameparams.windowwidth - obj.radius]
            movement_space_y = [obj.radius, gameparams.windowheight - obj.radius]

            if obj.pos[0] <= movement_space_x[0] or obj.pos[0] >= movement_space_x[1]:
                obj.vel[0] = 0
                if obj.pos[0] <= movement_space_x[0]:
                    obj.pos[0] = movement_space_x[0]
                if obj.pos[0] >= movement_space_x[1]:
                    obj.pos[0] = movement_space_x[1]
            if obj.pos[1] <= movement_space_y[0] or obj.pos[1] >= movement_space_y[1]:
                obj.vel[1] = 0
                if obj.pos[1] <= movement_space_y[0]:
                    obj.pos[1] = movement_space_y[0]
                if obj.pos[1] >= movement_space_y[1]:
                    obj.pos[1] = movement_space_y[1]
        else:
            movement_space_x = [gameparams.pitchcornerx + obj.radius,
                                gameparams.pitchcornerx + gameparams.pitchwidth - obj.radius]
            movement_space_y = [gameparams.pitchcornery + obj.radius,
                                gameparams.pitchcornery + gameparams.pitchheight - obj.radius]

            if obj.pos[0] <= movement_space_x[0] or obj.pos[0] >= movement_space_x[1]:
                if obj.pos[1] >= gameparams.goaly[0] and obj.pos[1] <= gameparams.goaly[1]:
                    pass
                else:
                    obj.vel[0] = - 0.5 * obj.vel[0]
                    if obj.pos[0] <= movement_space_x[0]:
                        obj.pos[0] = movement_space_x[0] + (movement_space_x[0] - obj.pos[0]) / 2

                    if obj.pos[0] >= movement_space_x[1]:
                        obj.pos[0] = movement_space_x[1] + (movement_space_x[1] - obj.pos[0]) / 2

            if obj.pos[1] <= movement_space_y[0] or obj.pos[1] >= movement_space_y[1]:
                obj.vel[1] = - 0.5 * obj.vel[1]
                if obj.pos[1] <= movement_space_y[0]:
                    obj.pos[1] = movement_space_y[0] + (movement_space_y[0] - obj.pos[1]) / 2
                if obj.pos[1] >= movement_space_y[1]:
                    obj.pos[1] = movement_space_y[1] + (movement_space_y[1] - obj.pos[1]) / 2

    def makeEntityHitBall(self, obj, ball):
        if self.printDebug:
            print("HIT!!")
        # Updates the ball's velocity since a kick call was called from obj to ball
        ball.vel = ball.vel + gameparams.kickstrength * ball.inv_mass * obj.getDirectionTo(ball)
        return

    def resolveCollision(self, obj1, obj2, is_obj1_static = 0):
        # if there is a collision between the two objects, resolve it. Assumes two circles
        # Has flag for the case where obj2 is static and doesn't get any momentum
        direction = (obj1.pos - obj2.pos)
        distance = (np.linalg.norm(direction))

        # if the objects aren't overlapping, don't even bother resolving
        if distance > obj1.radius + obj2.radius:
            return

        # calculates normal and tangent vectors
        collisionnormal = direction / distance
        collisiontangent = np.array([direction[1], - direction[0]]) / (np.linalg.norm(direction))

        bouncingq = obj1.bouncingquotient * obj2.bouncingquotient
        if is_obj1_static == False:
            centerofmass = (obj1.pos * obj1.mass + obj2.pos * obj2.mass) / (obj1.mass + obj2.mass)

            # updates object components
            obj1normalvelocity = np.dot(np.array(obj1.vel), collisionnormal)
            obj2normalvelocity = np.dot(np.array(obj2.vel), collisionnormal)

            # inelastic collision formula
            obj1newnormalvelocity = (bouncingq * obj2.mass * (obj2normalvelocity - obj1normalvelocity) + obj1.mass * obj1normalvelocity + obj2.mass * obj2normalvelocity) / (obj1.mass + obj2.mass)
            obj2newnormalvelocity = (bouncingq * obj1.mass * (obj1normalvelocity - obj2normalvelocity) + obj2.mass * obj2normalvelocity + obj1.mass * obj1normalvelocity) / (obj2.mass + obj1.mass)
            obj1tangentvelocity = np.dot(np.array(obj1.vel), collisiontangent)
            obj2tangentvelocity = np.dot(np.array(obj2.vel), collisiontangent)

            obj1.vel = obj1newnormalvelocity * np.array(collisionnormal) + obj1tangentvelocity * np.array(collisiontangent)
            obj2.vel = obj2newnormalvelocity * np.array(collisionnormal) + obj2tangentvelocity * np.array(collisiontangent)

            obj1.pos = centerofmass + ((obj1.radius + obj2.radius) + bouncingq * (obj1.radius + obj2.radius - distance)) * collisionnormal * obj2.mass / (obj1.mass + obj2.mass)
            obj2.pos = centerofmass - ((obj1.radius + obj2.radius) + bouncingq * (obj1.radius + obj2.radius - distance)) * collisionnormal * obj1.mass / (obj1.mass + obj2.mass)
        else:
            # updates obj2 components since that's the only moving part
            obj1normalvelocity = np.dot(np.array(obj1.vel), collisionnormal)
            obj2normalvelocity = np.dot(np.array(obj2.vel), collisionnormal)
            velocityafter = (obj1normalvelocity + obj2normalvelocity) * bouncingq * 2

            obj1tangentvelocity = np.dot(np.array(obj1.vel), collisiontangent)
            obj2tangentvelocity = np.dot(np.array(obj2.vel), collisiontangent)

            obj1.vel = - velocityafter * np.array(collisionnormal) + obj1tangentvelocity * np.array(collisiontangent)
            obj2.vel = velocityafter * np.array(collisionnormal) + obj2tangentvelocity * np.array(collisiontangent)

            obj2.pos = obj1.pos - collisionnormal * (obj1.radius + obj2.radius)

    def detectAndResolveCollisions(self):
        # Handle ALL the collision in the sim, including borders, entities etc.

        # blocks the players that aren't kicking off from entering the centre/other half
        if self.has_the_game_been_kicked_off == False:
            if self.red_last_goal == True:
                for i in range(len(self.reds)):
                    player = self.reds[i]
                    if player.pos[0] >= gameparams.windowwidth // 2 - player.radius:
                        player.vel[0] = 0
                        player.pos[0] = gameparams.windowwidth // 2 - player.radius

                    self.keepOutOfCentre(self.reds[i])
            else:
                for i in range(len(self.blues)):
                    player = self.blues[i]
                    if player.pos[0] <= gameparams.windowwidth // 2 + player.radius:
                        player.vel[0] = 0
                        player.pos[0] = gameparams.windowwidth // 2 + player.radius

                    self.keepOutOfCentre(self.blues[i])

        # Keep all the players within the playing field
        for player in self.players:
            self.keepEntityInMovementSpace(player, 0)
        # And same for the balls. Keep in mind they have a different field size
        for ball in self.balls:
            self.keepEntityInMovementSpace(ball, 1)

        # Handle moving object - moving object collisions
        for i in range(len(self.moving_objects)):
            for j in range(i + 1, len(self.moving_objects)):
                self.resolveCollision(self.moving_objects[i], self.moving_objects[j])

        # Handle moving object - goal post collisions
        for thing in self.moving_objects:
            for goalpost in self.goalposts:
                self.resolveCollision(goalpost, thing, 1)

        # Handle ball kicks
        for player in self.players:
            for ball in self.balls:
                if player.getDistanceTo(ball) <= player.radius + ball.radius + 4:
                    self.has_the_game_been_kicked_off = True

                    if player.current_action.isKicking() and player.can_kick:
                        self.makeEntityHitBall(player, ball)
                        player.can_kick = False
            if not player.current_action.isKicking():
                player.can_kick = True

    def updatePositions(self):
        # Update all the positions of the entities, no collision detection
        for entity in self.moving_objects:
            entity.updatePosition()

    def resetMap(self):
        # Reset the positions of the entities in the sim
        for obj in self.moving_objects:
            obj.reset()
        if self.enforce_kickoff:
            self.has_the_game_been_kicked_off = False
        # TODO: Reset the game data?
        return

    def updateScore(self):
        # TODO: Fuck this.
        for ball in self.balls:
            if ball.pos[0] <= gameparams.pitchcornerx:
                self.blue_score += 1
                self.red_last_goal = False
                self.was_point_scored = True

                # TODO: Game data stuff?

                frames = 0
                if self.blue_score + self.red_score > 2:
                    # TODO: New match? Update the agents?
                    pass
                self.resetMap()
            elif ball.pos[0] >= gameparams.windowwidth - gameparams.pitchcornerx:
                self.red_score += 1
                self.red_last_goal = True
                self.was_point_scored = True

                # TODO: Game data stuff?

                frames = 0
                if self.blue_score + self.red_score > 2:
                    # TODO: New match? Update the agents?
                    pass
                self.resetMap()
        return

    def checkGoals(self):
        #Checks all the balls, returns tuple of (red scores, blue scores)
        countedGoals = (0,0)
        for ball in self.balls:
            if ball.pos[0] <= gameparams.pitchcornerx:
                countedGoals[1] += 1
            elif ball.pos[0] >= gameparams.windowwidth - gameparams.pitchcornerx:
                countedGoals[0] += 1
        return countedGoals

    def getFeedback(self):
        # TODO: Idk in what form you want this to be, can be easily modified.
        if self.printDebug:
            # Print some stuff
            print("Frame {}, score R-B: {}-{}".format(self.frames, self.red_score, self.blue_score))
            if self.was_point_scored:
                print("    A point was scored, nice!")
            for obj in self.reds:
                print("    red player at: {:.3f}; {:.3f} with velocity {:.3f}; {:.3f}".format(obj.pos[0], obj.pos[1], obj.vel[0], obj.vel[1]))
            for obj in self.blues:
                print("    blue player at: {:.3f}; {:.3f} with velocity {:.3f}; {:.3f}".format(obj.pos[0], obj.pos[1], obj.vel[0], obj.vel[1]))
            for obj in self.balls:
                print("    ball at: {:.3f}; {:.3f} with velocity {:.3f}; {:.3f}\n".format(obj.pos[0], obj.pos[1], obj.vel[0], obj.vel[1]))
<<<<<<< HEAD
=======

>>>>>>> db5dca5c06241018f549d78f81d512d447b8dd2c
        return

    def getState(self, format = "positions"):
        # Returns data based off of format specified
        # TODO: come up with better name than format

        # Positions give list of player positions, then ball positions
        # All positions are just tuples (x,y)
        # First element is list of red player positions
        # Second element is list of blue player positions
        # Thirst is list of ball positions
        if format == "positions":
            redsPos = [ rPlayer.pos for rPlayer in self.reds ]
            bluesPos = [ bPlayer.pos for bPlayer in self.blues ]
            ballsPos = [ ball.pos for ball in self.balls ]

            return (redsPos, bluesPos, ballsPos )
        elif format == "full info":
            redsInfo = [ [rPlayer.pos, rPlayer.vel, rPlayer.current_action.isKicking() and rPlayer.can_kick] for rPlayer in self.reds ]
            bluesInfo = [ [bPlayer.pos, bPlayer.vel, bPlayer.current_action.isKicking() and bPlayer.can_kick] for bPlayer in self.blues ]
            ballsInfo = [ [ball.pos, ball.vel ] for ball in self.balls ]

            return (redsInfo, bluesInfo, ballsInfo )

        elif format == "state-action pairs":
            state = [ [object.pos, object.vel ] for object in self.moving_objects]
            action = [ [player.current_action] for player in self.players]
            return (state, action)

        elif format == "raw sa pairs":
            state = [ [object.pos, object.vel ] for object in self.moving_objects]
            action = [ player.current_action.rawAction() for player in self.players]
            return [state, action]



    def giveCommands(self, actions, actionFormat = "raw"):
        # Gives commands to all the controllable entities in the game in the form of a list pf commands.
        # Each command is a tuple of size 2 specifying direction (18 possible states) and then the kick state.
        # The position of the command in the list determines which entity the command is sent to.
        # TODO: Pls complete this function

        # NOTE: reds come before blues.
        if actionFormat == "raw":
            for i in range(len(self.players)):
                self.players[i].current_action = playeraction.Action(actions[i][0], actions[i][1])
        elif actionFormat == "object":
            for i in range(len(self.players)):
                self.players[i].current_action = actions[i]
        return

    def step(self):
        self.frames += 1
        self.was_point_scored = 0
        # TODO: pygame clock

        # Update positions
        self.updatePositions()

        # Hande collisions
        self.detectAndResolveCollisions()

        # Update the score of the game
        if self.auto_score:
            self.updateScore()

        ### TODO: Handle key events
        ### keys = pygame.key.get_pressed()
        return
