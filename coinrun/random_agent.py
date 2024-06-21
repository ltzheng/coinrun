import numpy as np
from coinrun import setup_utils, make
import os
from tqdm import tqdm
from PIL import Image


def random_agent(file_stem, num_episodes=10000, max_steps=1000):
    # remove existing dataset
    if os.path.exists(file_stem):
        os.system(f"rm -r {file_stem}")
    os.makedirs(os.path.join(file_stem, "dataset/images"))
    setup_utils.setup_and_load()

    for i in tqdm(range(num_episodes)):
        metadata = []
        print(f"Episode {i}")
        last_acts = None
        env = make('standard', num_envs=1)

        env.step(np.array([0]))
        obs = env.get_images()
        image = Image.fromarray(obs[0])
        img_filename = f"episode_{i}_step_0.jpg"
        image.save(os.path.join(file_stem, "dataset/images", img_filename))
        metadata.append({"obs": img_filename})

        for step in range(max_steps):
            acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            # no action repeat
            if last_acts is not None:
                while last_acts == acts[0]:
                    acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            last_acts = acts[0]
            _, rews, dones, _ = env.step(acts)  # obs shape (num_envs, 64, 64, 3)
            obs = env.get_images()
            image = Image.fromarray(obs[0])
            img_filename = f"episode_{i}_step_{step + 1}.jpg"
            image.save(os.path.join(file_stem, "dataset/images", img_filename))

            dict_to_add = {
                "acts": acts[0].tolist(),
                "obs": img_filename,
                "rews": rews[0].tolist(),
                "dones": dones[0],
            }
            metadata.append(dict_to_add)

        env.close()

        with open(os.path.join(file_stem, f"dataset/metadata.jsonl"), "a") as f:
            f.write(f"{metadata}\n")


if __name__ == '__main__':
    file_stem = "/media/longtao/DATA/Genie/"
    random_agent(file_stem=file_stem)
