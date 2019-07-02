import playeraction
import entities
import playeraction


class GameSim:
    def __init__(self, red_players, blue_players, balls):
        # Intialise the entities
        self.reds = red_players
        self.blues = blue_players
        self.balls = balls
        self.goalposts = [entities.GoalPost(np.array(pitchcornerx, goalcornery)),
                          entities.GoalPost(np.array(pitchcornerx, goalcornery + goalsize)),
                          entities.GoalPost(np.array(windowwidth - pitchcornerx, goalcornery)),
                          entities.GoalPost(np.array(windowwidth - pitchcornerx, goalcornery + goalsize))]
        self.centre_block = entities.CentreCircleBlock(np.array(gameparams.ballstart))

        # Create useful groupings
        self.players = self.reds + self.blues
        self.moving_objects = self.players + self.balls

        # Game state flags
        self.has_the_game_been_kicked_off = True

        self.red_last_goal = False

    def resolveCollision(self, obj1, obj2, is_obj1_static = 0):
        # if there is a collision between the two objects, resolve it. Assumes two circles
        # Has flag for the case where obj2 is static and doesn't get any momentum
        direction = (obj1.pos - obj2.pos)
        distance = (np.linalg.norm(direction))
        bouncingq = obj1.bouncingquotient * obj2.bouncingquotient
        centerofmass = (obj1.pos * obj1.mass + obj2.pos * obj2.mass) / (obj1.mass + obj2.mass)

        # if the objects aren't overlapping, don't even bother resolving
        if distance > obj1.radius + obj2.radius:
            return

        # calculates normal and tangent vectors
        collisionnormal = direction / distance
        collisiontangent = np.array([direction[1], - direction[0]]) / (np.linalg.norm(direction))

        if !is_obj1_static:
            # updates object components
            obj1normalvelocity = np.dot(np.array(obj1.velocity), collisionnormal)
            obj2normalvelocity = np.dot(np.array(obj2.velocity), collisionnormal)

            # inelastic collision formula
            obj1newnormalvelocity = (bouncingq * obj2.mass * (obj2normalvelocity - obj1normalvelocity) + obj1.mass * obj1normalvelocity + obj2.mass * obj2normalvelocity) / (obj1.mass + obj2.mass)
            obj2newnormalvelocity = (bouncingq * obj1.mass * (obj1normalvelocity - obj2normalvelocity) + obj2.mass * obj2normalvelocity + obj1.mass * obj1normalvelocity) / (obj2.mass + obj1.mass)
            obj1tangentvelocity = np.dot(np.array(obj1.velocity), collisiontangent)
            obj2tangentvelocity = np.dot(np.array(obj2.velocity), collisiontangent)

            obj1.velocity = obj1newnormalvelocity * np.array(collisionnormal) + obj1tangentvelocity * np.array(collisiontangent)
            obj2.velocity = obj2newnormalvelocity * np.array(collisionnormal) + obj2tangentvelocity * np.array(collisiontangent)

            obj1.pos = centerofmass + ((obj1.radius + obj2.radius) + bouncingq * (obj1.radius + obj2.radius - distance)) * collisionnormal * obj2.mass / (obj1.mass + obj2.mass)
            obj2.pos = centerofmass - ((obj1.radius + obj2.radius) + bouncingq * (obj1.radius + obj2.radius - distance)) * collisionnormal * obj1.mass / (obj1.mass + obj2.mass)
        else:
            # updates obj2 components since that's the only moving part
            obj1normalvelocity = np.dot(np.array(obj1.velocity), collisionnormal)
            obj2normalvelocity = np.dot(np.array(obj2.velocity), collisionnormal)
            velocityafter = (obj1normalvelocity + obj2normalvelocity) * bouncingq * 2

            obj1tangentvelocity = np.dot(np.array(obj1.velocity), collisiontangent)
            obj2tangentvelocity = np.dot(np.array(obj2.velocity), collisiontangent)

            obj1.velocity = - velocityafter * np.array(collisionnormal) + obj1tangentvelocity * np.array(collisiontangent)
            obj2.velocity = velocityafter * np.array(collisionnormal) + obj2tangentvelocity * np.array(collisiontangent)

            obj2.pos = obj1.pos - collisionnormal * (obj1.radius + obj2.radius)

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

    def giveCommands(self, actions):
        # Gives commands to all the controllable entities in the game in the form of a list pf commands.
        # Each command is a tuple of size 2 specifying kick state and direction (18 possible states).
        # The position of the command in the list determines which entity the command is sent to.
        continue

    def getFeedback(self):
        # Idk in what form you want this to be, can be easily modified.
        continue

    def keepEntityInMovementSpace(self, obj, is_ball = 0):
        # should keep things on the board where the movement happens

        if !is_ball:
            movement_space_x = [obj.radius, gameparams.windowwidth - obj.radius]
            movement_space_y = [obj.radius, gameparams.windowheight - obj.radius]

            if obj.pos[0] <= movement_space_x[0] or obj.pos[0] >= movement_space_x[1]:
                obj.velocity[0] = 0
                if obj.pos[0] <= movement_space_x[0]:
                    obj.pos[0] = movement_space_x[0]
                if obj.pos[0] >= movement_space_x[1]:
                    obj.pos[0] = movement_space_x[1]
            if obj.pos[1] <= movement_space_y[0] or obj.pos[1] >= movement_space_y[1]:
                obj.velocity[1] = 0
                if obj.pos[1] <= movement_space_y[0]:
                    obj.pos[1] = movement_space_y[0]
                if obj.pos[1] >= movement_space_y[1]:
                    obj.pos[1] = movement_space_y[1]
        else:
            movement_space_x = [gameparams.pitchcornerx + obj.radius,
                                gameparams.pitchcornerx + gameparams.pitchwidth - obj.radius]
            movement_space_y = [gameparams.pitchcornery + obj.radius,
                                gameparams.pitchcornerx + gameparams.pitchheight - obj.radius]

            if obj.pos[0] <= movement_space_x[0] or obj.pos[0] >= movement_space_x[1]:
                if obj.pos[1] >= gameparams.goaly[0] and obj.pos[1] <= gameparams.goaly[1]:
                    pass
                else:
                    obj.velocity[0] = - 0.5 * ball.velocity[0]
                    if obj.pos[0] <= movement_space_x[0]:
                        obj.pos[0] = movement_space_x[0] + (movement_space_x[0] - obj.pos[0]) / 2

                    if ball.pos[0] >= movement_space_x[1]:
                        obj.pos[0] = movement_space_x[1] + (movement_space_x[1] - obj.pos[0]) / 2

            if obj.pos[1] <= movement_space_y[0] or obj.pos[1] >= movement_space_y[1]:
                obj.velocity[1] = - 0.5 * obj.velocity[1]
                if obj.pos[1] <= movement_space_y[0]:
                    obj.pos[1] = movement_space_y[0] + (movement_space_y[0] - obj.pos[1]) / 2
                if obj.pos[1] >= movement_space_y[1]:
                    obj.pos[1] = movement_space_y[1] + (movement_space_y[1] - obj.pos[1]) / 2

    def detectAndResolveCollisions(self):
        # blocks the players that aren't kicking off from entering the centre/other half
        if self.has_the_game_been_kicked_off == False:
            if self.red_last_goal == True:
                for i in range(len(reds)):
                    if reds[i].pos[0] >= windowwidth // 2 - playerradius:
                        reds[i].velocity[0] = 0
                        reds[i].pos[0] = windowwidth // 2 - playerradius

                    self.keepoutofcentre(reds[i])
            else:
                for i in range(len(blues)):
                    if blues[i].pos[0] <= windowwidth // 2 + playerradius:
                        blues[i].velocity[0] = 0
                        blues[i].pos[0] = windowwidth // 2 + playerradius

                    self.keepoutofcentre(blues[i])

        # Keep all the players within the playing field
        for player in players:
            self.keepEntityInMovementSpace(player, 0)
        # And same for the balls. Keep in mind they have a different field size
        for ball in balls:
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
                if 
        return

    def moveEntities(self):
        for entity in self.moving_objects:
            entity.updatePosition()
        return


    def step(self):
        # TODO: pygame clock

        # Update positions
        self.moveEntities()

        # Hande collisions
        self.detectAndResolveCollisions()

        ### handles the key events
        ### keys = pygame.key.get_pressed()
