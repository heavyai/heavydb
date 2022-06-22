#!/usr/bin/python3

import subprocess, sys, os.path
import statistics, json
import matplotlib.pyplot as plt

NUM_TEST = 4
JSON_NAME = 'perf.json'


def parse_arguments():
    cmd = sys.argv[1:]
    num_runs = cmd[0]
    if num_runs[-1] == 'x':
        num_runs = int(num_runs[:-1])
        cmd.pop(0)
    else:
        num_runs = 1
    return num_runs, cmd


def do_runs(num_runs, cmd):
    time = [[] for _ in range(NUM_TEST)]
    cpu = [[] for _ in range(NUM_TEST)]

    for i in range(num_runs):
        r = subprocess.run(cmd, stdout = subprocess.PIPE)
        lines = r.stdout.decode('utf-8').splitlines()

        for j in range(NUM_TEST):
            r = lines[j - NUM_TEST].split()
            time[j].append(int(r[1]))
            cpu[j].append(int(r[3]))

    time = [statistics.median(r) for r in time]
    cpu = [statistics.median(r) for r in cpu]
    return time, cpu


def load_json():
    if not os.path.exists(JSON_NAME): return [[], []]

    with open(JSON_NAME) as f:
        data = json.load(f)

    return data


def draw_graphs(name, data):
    for i in range(NUM_TEST):
        d = [x[i] for x in data]
        n = name + str(i)
        plt.plot(d)
        plt.title(n)
        plt.xticks([])
        plt.savefig(n + '.pdf')
        plt.close()


def process_runs(perf_json, time, cpu):
    perf_json[0].append(time)
    perf_json[1].append(cpu)

    with open(JSON_NAME, 'w') as f:
        json.dump(perf_json, f)
    return perf_json


def draw_all(perf_json):
    draw_graphs('time', perf_json[0])
    draw_graphs('cpu', perf_json[1])


num_runs, cmd = parse_arguments()
time, cpu = do_runs(num_runs, cmd)
perf_json = load_json()
perf_json = process_runs(perf_json, time, cpu)
draw_all(perf_json)

