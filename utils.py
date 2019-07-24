import numpy as np
import time

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], type(np.array([]))):
        return flatten(S[0].tolist()) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def reverseAction(act):
    move, kick = act
    if move != 0:
        move = ((move + 3) % 8) + 1
    return move, kick

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.name_counters = {}

    def printTime(self):
        time_elapsed = time.time() - self.start_time
        print("Time elapsed: {:.3f}s".format(time_elapsed))

    def getElapsedTime(self):
        return time.time() - self.start_time

    def getLapDuration(self):
        cur_time = time.time()
        time_dur = cur_time - self.last_time
        self.last_time = cur_time
        return time_dur

    def logType(self, name, supress_print = True):
        lap_duration = self.getLapDuration()

        if name not in self.name_counters:
            self.name_counters[name] = lap_duration
        else:
            self.name_counters[name] += lap_duration
        if not supress_print:
            print("Time: {:.3f}s, lap: {:.3f}s, type_name = {}".format(self.getElapsedTime(), lap_duration, name))

    def printSummary(self):
        tot = 0
        for key, value in self.name_counters.items():
            tot += value
        for key, value in self.name_counters.items():
            print("type_name: {},\t percentage: {:.3f}%".format(key, value/tot * 100))

    def reset(self):
        self.start_time = time.time()
        self.last_time = time.time()
        self.name_counters = {}

global global_timer
global_timer = Timer()
