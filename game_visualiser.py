import argparse
import os

import torch

from agents import ACagent, humanACagent, humanagent, idleagent
from game_displayer import basicdisplayer
from game_simulator import gameparams as gp
from game_simulator import gamesim
from game_log import log
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
    '--red-normalised',
    action="store_true",
    help='Specify whether the model of the red team takes normalised input or not')
parser.add_argument(
    '--blue-normalised',
    action="store_true",
    help='Specify whether the model of the blue team takes normalised input or not')
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
parser.add_argument(
    '--save-master-dir',
    type=str,
    default = "recorded_games",
    help='Specify the subdirectory where all the recorded games folders are gonna be saved in')
parser.add_argument(
    '--save-dir',
    type=str,
    default = "None",
    help='Specify the name of the directory where the current sessions is gonna be saved in')
parser.add_argument(
    '--save-step-length',
    type=int,
    default = -1,
    help='Specify the saving step length. Defaults to step length')
parser.add_argument(
    '--max-games',
    type=int,
    default = 2147483648,
    help='Specify the number of games to be played. Defaults to infinity')
args = parser.parse_args()


default_red_bindings = ('w', 'd', 's', 'a', 'LSHIFT')
default_blue_bindings = ('UP', 'RIGHT', 'DOWN', 'LEFT', 'u')

def getAgents(display, red_debug_surf, blue_debug_surf):
    if args.red_model != None:
        model_red = torch.load(args.load_dir + args.red_model + ".model").to("cpu")
        agent_red_ = ACagent.ACAgent(model_red, "red", "random", red_debug_surf, args.red_normalised)
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
        model_blue = torch.load(args.load_dir + args.blue_model + ".model").to("cpu")
        agent_blue_ = ACagent.ACAgent(model_blue, "blue", "random", blue_debug_surf, args.blue_normalised)
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
    for game_number in range(args.max_games):
        exit_loop = False
        # Initialise the game logger if needed
        if args.save_dir != "None":
            game_logger = log.Game()
            if args.save_step_length == -1:
                args.save_step_length = args.step_length
            save_counter = 0

            if not os.path.exists(f"{args.save_master_dir}"):
                os.makedirs(f"{args.save_master_dir}")

        while True:
            game_ended = False

            # Query each agent on what commands should be sent to the game simulator
            if display != None:
                display.getInput()
            game.giveCommands([a.getAction(game.log()) for a in agents])

            for i in range(args.step_length):
                game_ended = game_ended or game.step()

                # Append a frame to the game_logger if enabled
                if args.save_dir != "None":
                    save_counter += 1
                    if save_counter == args.save_step_length:
                        save_counter = 0
                        game_logger.append(game.log())

                if game_ended:
                    # Save the game logger data if enabled
                    if args.save_dir != "None":
                        if not os.path.exists(f"{args.save_master_dir}/{args.save_dir}"):
                            os.makedirs(f"{args.save_master_dir}/{args.save_dir}")
                        game_logger.save(f"{args.save_master_dir}/{args.save_dir}/{game_number}")
                    break

            if game_ended:
                break

            if display != None:
                # Update the graphical interface canvas
                display.drawFrame(game.log())

                if display.rip:
                    display.shutdown()
                    exit_loop = True
                    break
        if exit_loop:
            break

if __name__ == "__main__":
    main()
