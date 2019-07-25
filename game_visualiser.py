#! /usr/bin/python

import argparse

import torch

from agents import ACagent, humanACagent, humanagent, idleagent
from game_displayer import basicdisplayer
from game_simulator import gameparams as gp
from game_simulator import gamesim
from move_displayer import movedisplayer

# Only works with 1v1 so far
# Example command line code:
#   python game_visualiser.py --red-model="trained_fixed_v1" --blue-model="arun_v4" --suppress-display

parser = argparse.ArgumentParser(description='Visualise different matchups between agents')
parser.add_argument(
    '--seed', type=int, default=-1, help='random seed (default: -1)')
parser.add_argument(
    '--red-human',
    action="store_true",
    help='Does the red team have a human player?')
parser.add_argument(
    '--blue-human',
    action="store_true",
    help='Does the blue team have a human player?')
parser.add_argument(
    '--load-dir',
    default='models/',
    help='directory to load the agents from (default models/)')
parser.add_argument(
    '--red-model',
    default=None,
    help='Specify the model of the red team if there is any')
parser.add_argument(
    '--blue-model',
    default=None,
    help='Specify the model of the blue team if there is any')
parser.add_argument(
    '--print-debug',
    action="store_true",
    help='Specify whether debug info should be printed or not')
parser.add_argument(
    '--not-auto-score',
    dest="auto_score",
    action="store_false",
    help='Specify whether the game should autoscore goals')
parser.add_argument(
    '--not-rand-reset',
    dest="rand_reset",
    action="store_false",
    help='Specify whether gamesim places entities randomly when reseting')
parser.add_argument(
    '--suppress-scorekeeping',
    action="store_true",
    help='Specify whether gamesim should print scores to terminal')
parser.add_argument(
    '--suppress-display',
    action="store_true",
    help='Specify whether the display of the game should be suppressed or not')
parser.add_argument(
    '--max-steps',
    type=int,
    default = -1,
    help='Specify the maximum number of steps in gamesim per game')
parser.add_argument(
    '--step-length',
    type=int,
    default = 1,
    help='Specify the number of steps per action received from the models')
args = parser.parse_args()


default_red_bindings = ('w', 'd', 's', 'a', 'LSHIFT')
default_blue_bindings = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'u')

def getAgents(display, red_debug_surf, blue_debug_surf):
    if args.red_model != None:
        model_red = torch.load(args.load_dir + args.red_model + ".model")
        agent_red_ = ACagent.ACAgent(model_red, "red", "random", red_debug_surf, False)
        if args.red_human == True:
            agent_red = humanACagent.HumanACAgent(default_red_bindings, display, agent_red_)
        else:
            agent_red = agent_red_
    else:
        if args.red_human == True:
            agent_red = humanagent.HumanAgent(default_red_bindings, display)
        else:
            agent_red = idleagent.IdleAgent()

    if args.blue_model != None:
        model_blue = torch.load(args.load_dir + args.blue_model + ".model")
        agent_blue_ = ACagent.ACAgent(model_blue, "blue", "random", blue_debug_surf, False)
        if args.blue_human == True:
            agent_blue = humanACagent.HumanACAgent(default_blue_bindings, display, agent_blue_)
        else:
            agent_blue = agent_blue_
    else:
        if args.blue_human == True:
            agent_blue = humanagent.HumanAgent(default_blue_bindings, display)
        else:
            agent_blue = idleagent.IdleAgent()

    return (agent_red, agent_blue)

def main():
    if args.suppress_display and (args.red_human or args.blue_human):
        raise ValueError("Human players need display to function")

    red_debug_surf = movedisplayer.DebugSurf()
    blue_debug_surf = movedisplayer.DebugSurf()

    if not args.suppress_display:
        display = basicdisplayer.GameWindow(gp.windowwidth + 2 * 256, gp.windowheight,\
                                        debug_surfs = [red_debug_surf.surf, blue_debug_surf.surf])
    else:
        display = None
    agents = getAgents(display, red_debug_surf, blue_debug_surf)

    game = gamesim.GameSim(1, 1, 1, printDebug = args.print_debug, print_score_update = not args.suppress_scorekeeping,auto_score = args.auto_score, rand_reset = args.rand_reset, max_steps = args.max_steps)

    # Run the game
    while True:
        # Query each agent on what commands should be sent to the game simulator
        game.giveCommands([a.getAction(game.log()) for a in agents])

        for i in range(args.step_length):
            game.step()

        if display != None:
            # Update the graphical interface canvas
            display.drawFrame(game.log())

            display.getInput()

            if display.rip:
                display.shutdown()
                break

if __name__ == "__main__":
    main()
