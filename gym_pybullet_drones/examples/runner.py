import json
from ro47005 import run


if __name__ == "__main__":
    output_list = []
    for seed in range(100):
        try:
            output = run(seed, show_rrt_graphs=False, plot=False, print_drone_info=False)
            output["seed"] = seed
            output_list.append(output)
        except:
            print(f"error on seed {seed}")

    with open("runner.json", mode="w") as f:
        f.write(json.dumps(output_list))
else:
    print("ERROR")

