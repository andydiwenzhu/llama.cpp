import subprocess
import sys
import time

import pandas as pd

max_context = 480

model_paths = {
    "70b": "../ordered-llama-2-70b-chat-hf-q4_0.gguf",
    "13b": "../ordered-llama-2-13b-chat-hf-fp16.gguf",
}

model_offloads = {
    "70b": [8, 16],
    "13b": [4, 8, 6, 12],
}

chunk_sizes = {
    "70b": 48,
    "13b": 30,
}

model_decode = {
    "70b": 2,
    "13b": 8,
}

def run_exp1(model, prefix, chunk):
    prompt_len = prefix + chunk
    prompt = " ".join(["good"] * (prompt_len-1))
    if prefix == 0:
        command = ' '.join(['make -j && ./main', '-m', model_paths[model], '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', str(model_offloads[model][1]+1)])
    else:
        command = ' '.join(['make -j && ./main', '-m', model_paths[model], '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', str(model_offloads[model][1]+1), '-b', str(prefix)])
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
                x = run_exp1(model, c * p, c)
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


def run_exp2(model, prefix, chunk, mem):
    prompt_len = prefix + chunk
    prompt = " ".join(["good"] * (prompt_len-1))
    if prefix == 0:
        command = ' '.join(['make -j && ./main', '-m', model_paths[model], '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', str(model_offloads[model][mem]+1)])
    else:
        command = ' '.join(['make -j && ./main', '-m', model_paths[model], '-p', f'"{prompt}"', '-n', '1', '-e', '--threads', str(model_offloads[model][mem]+1), '-b', str(chunk)])
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    r = str(stderr)
    key = "prompt eval time ="
    r = r[r.rfind(key) + len(key):]
    r = r[:r.find("ms")]
    x = float(r.strip())
    return x / 1000.0

def eval_exp2(model, mem=1):
    res = []
    for c in [chunk_sizes[model]]:
        for p in range(max_context // 2 // c):
            for j in range(2):
                x = run_exp2(model, c * p, c, mem)
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
                r.to_csv(f"../results/exp2_{model}-offload-{model_offloads[model][mem]}-layer.csv", index=False)



def eval_exp3(model, stage="prefill"):
    c = chunk_sizes[model]
    if stage == "prefill":
        windows = [int(c * (i/4)) for i in range(1,9)]
    else:
        assert(stage == "decode")
        windows = [model_decode[model]]
    res = []
    for w in windows:
        for p in range(int(stage == "decode"), max_context // max(w, c) + int(stage == "decode")):
            for j in range(2):
                x = run_exp1(model, c * p, w)
                res.append({
                    "model": model,
                    "chunk": w,
                    "prefix": c * p,
                    "total_ctx": c * p + w,
                    "layer_time": x,
                    "8g_total_time": x * model_offloads[model][0],
                    "16g_total_time": x * model_offloads[model][1],
                    "round": j,
                })
                time.sleep(5)
                r = pd.DataFrame(res)
                print(r)
                r.to_csv(f"../results/exp3_{model}-{stage}-offload-{model_offloads[model][1]}-layer.csv", index=False)


if __name__ == '__main__':
    #eval_exp1("70b")
    #eval_exp2("70b", 0)
    #eval_exp2("70b", 1)
    #eval_exp3("70b", "prefill")
    #eval_exp3("70b", "decode")
    #eval_exp2("13b", 0)
    #eval_exp2("13b", 1)
    #eval_exp2("13b", 2)
    #eval_exp2("13b", 3)
    #eval_exp3("13b", "prefill")
    eval_exp3("13b", "decode")
