import numpy as np
from coinrun import setup_utils, make
import os
from PIL import Image
from tqdm import tqdm


def random_agent(num_episodes=10000, max_steps=1000):
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/images", exist_ok=True)
    setup_utils.setup_and_load()

    num_envs = 100
    assert num_episodes % num_envs == 0
    num_iters = num_episodes // num_envs

    for i in tqdm(range(num_iters), desc="Iter"):
        metadata = [[] for _ in range(num_envs)]
        print(f"Episode {i * num_envs}-{(i + 1) * num_envs}")
        last_acts = None
        env = make('standard', num_envs=num_envs)

        env.step(np.array([0] * num_envs))
        obs = env.get_images()
        for j in range(num_envs):
            image = Image.fromarray(obs[j])
            image.save(f"dataset/images/episode_{i + j}_step_0.jpg")
            metadata[j].append({"obs": f"dataset/images/episode_{i + j}_step_0.jpg"})

        for step in tqdm(range(max_steps), desc="Steps"):
            acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            # no action repeat
            if last_acts is not None:
                for j in range(num_envs):
                    while last_acts[j] == acts[j]:
                        acts[j] = env.action_space.sample()
            last_acts = acts
            _, rews, dones, _ = env.step(acts)  # obs shape (num_envs, 64, 64, 3)
            obs = env.get_images()

            for j in range(num_envs):
                image = Image.fromarray(obs[j])
                image.save(f"dataset/images/episode_{i + j}_step_{step + 1}.jpg")
                dict_to_add = {
                    "acts": acts[j].tolist(),
                    "obs": f"images/episode_{i + j}_step_{step + 1}.jpg",
                    "rews": rews[j].tolist(),
                    "dones": dones[j],
                }
                metadata[j].append(dict_to_add)

        env.close()

        with open(f"dataset/metadata.jsonl", "a") as f:
            for m in metadata:
                f.write(f"{m}\n")


if __name__ == '__main__':
    random_agent()