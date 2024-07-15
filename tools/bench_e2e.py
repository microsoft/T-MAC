import subprocess
import argparse
import regex as re
import csv


def parse_tps(output: str):
    for line in output.splitlines():
        try:
            col = line.split("|")[-2]
            pattern = r"([\d.]+) ± ([\d.]+)"
            m = re.search(pattern, col)
            if m:
                return float(m[1]), float(m[2])
        except IndexError:
            continue


def run(num_threads):
    command = [
        FLAGS.bench_path,
        "-m", f"{FLAGS.model_path}",
        "-n", f"{FLAGS.n_tokens}",
        "-ngl", "0",
        "-b", "1",
        "-t", f"{num_threads}",
        "-p", "0",
    ]
    return parse_tps(subprocess.check_output(command).decode())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-b", "--bench_path", type=str)
    parser.add_argument("-n", "--n_tokens", type=int, default=128)
    parser.add_argument("-mt", "--max_threads", type=int)
    parser.add_argument("-o", "--output_csv", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = parse_args()

    num_threads_list = []
    # prevent overheating by reordering num_threads
    for t in range(1, (FLAGS.max_threads + 1) // 2 + 1):
        num_threads_list.append(t)
        if 1 + FLAGS.max_threads - t != num_threads_list:
            num_threads_list.append(1 + FLAGS.max_threads - t)
    with open(FLAGS.output_csv, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["num_threads", "mean", "std"])
        for t in num_threads_list:
            mean, std = run(t)
            writer.writerow([t, mean, std])
