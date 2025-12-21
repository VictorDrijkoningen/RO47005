import json
from ro47005 import run


error_list = list()

if __name__ == "__main__":
    output_list = []
    for seed in range(100):
        try:
            print(f"Starting run with seed: {seed}")
            output = run(seed, show_rrt_graphs=False, plot=False, print_debug_info=False)
            output["seed"] = seed
            output_list.append(output)
        except Exception as e:
            print(f"error on seed {seed}, exception: {e}")
            error_list.append(f"error on seed {seed}, exception: {e}")

    with open("runner.json", mode="w") as f:
        f.write(json.dumps(output_list))
else:
    print("ERROR")


print(error_list)
