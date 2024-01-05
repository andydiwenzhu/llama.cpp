import subprocess
import sys
import time

import pandas as pd

model_paths = {
    "70b": "../ordered-llama-2-70b-chat-hf-q4_0.gguf",
    "13b": "../ordered-llama-2-13b-chat-hf-fp16.gguf"
}


def run_exp1(model, prefix, chunk):
    prompt_len = prefix + chunk
    prompt = " ".join(["good"] * (prompt_len-1))
    if prefix == 0:
        command = ' '.join(['make -j && ./main', '-m', model, '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', '17'])
    else:
        command = ' '.join(['make -j && ./main', '-m', model, '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', '17', '-b', str(prefix)])
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    r = str(stderr)
    key = "[DEBUG] gpu_time ="
    r = r[r.rfind(key) + len(key):]
    r = r[:r.find(" seconds")]
    x = float(r.strip())
    return x

def eval_exp1(model):
    res = []
    for c in range(1, 129):
        for p in range(1): #int(127 / c) + 1):
            for j in range(2):
                x = run_exp1(model_paths[model], c * p, c)
                res.append({
                    "model": model,
                    "prefix/chunk": p, 
                    "chunk": c,
                    "prefix": c * p,
                    "total_ctx": c * p + c,
                    "time": x,
                    "round": j,
                })
                time.sleep(5)
                r = pd.DataFrame(res)
                print(r)
                r.to_csv(f"../results/exp1_{model}.csv", index=False)


def run_exp2(model, prefix, chunk, layer):
    prompt_len = prefix + chunk
    prompt = " ".join(["good"] * (prompt_len-1))
    if prefix == 0:
        command = ' '.join(['make -j && ./main', '-m', model, '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', str(layer)])
    else:
        command = ' '.join(['make -j && ./main', '-m', model, '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', str(layer), '-b', str(chunk)])
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    r = str(stderr)
    key = "prompt eval time ="
    r = r[r.rfind(key) + len(key):]
    r = r[:r.find("ms")]
    x = float(r.strip())
    return x / 1000.0

def eval_exp2(model, mem=16):
    res = []
    for c in [44, 46, 48]:
        for p in range(3):
            for j in range(3):
                x = run_exp2(model_paths[model], c * p, c, mem+1)
                res.append({
                    "model": model,
                    "prefix/chunk": p, 
                    "chunk": c,
                    "prefix": c * p,
                    "total_ctx": c + c * p,
                    "time": x,
                    "time/token": x / (c + c * p),
                    "token/sec": (c + c * p) / x,
                    "round": j,
                })
                time.sleep(5)
                r = pd.DataFrame(res)
                print(r)
                r.to_csv(f"../results/exp2_{model}-{mem}g.csv", index=False)



def eval_exp3(model):
    res = []
    for c in [2]:
        for p in range(1, 8):
            for j in range(2):
                x = run_exp1(model_paths[model], 64 * p, c)
                res.append({
                    "model": model,
                    "chunk": c,
                    "prefix": 64 * p,
                    "total_ctx": 64 * p + c,
                    "layer_time": x,
                    "total_time": x * 16,
                    "round": j,
                })
                time.sleep(5)
                r = pd.DataFrame(res)
                print(r)
                r.to_csv(f"../results/exp3_{model}-w2.csv", index=False)


if __name__ == '__main__':
    eval_exp1("70b")
    #eval_exp2("70b", 8)
    #eval_exp3("70b")