import subprocess
import sys
import time

import pandas as pd


def run(model='../ordered-llama-2-13b-chat-hf-q8_0.gguf', prompt_len=16):
    prompt = " ".join(["good"] * prompt_len)
    command = ' '.join(['./main', '-m', model, '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', '64'])
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    r = str(stderr)
    key = "prompt eval time ="
    r = r[r.find(key) + len(key):]
    r = r[:r.find("ms")]
    x = float(r.strip())
    return x / 1000.0


def eval(models=None):
    if models is None:
        mf = '../ordered-llama-2-{}-chat-hf-{}.gguf'
        models = [mf.format('13b', 'q8_0'), mf.format('70b', 'q2_k'), mf.format('70b', 'q4_0')]

    res = []
    for m in models:
        for i in range(4, 7):
            for j in range(6):
                x = run(m, 2 ** i)
                res.append({
                    "model": m,
                    "token": 2 ** i,
                    "time": x,
                    "round": j,
                })
                time.sleep(30)
                r = pd.DataFrame(res)
                print(r)
                r.to_pickle("../results/result_perf1.pkl")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        eval(sys.argv[1:])
    else:
        eval()
