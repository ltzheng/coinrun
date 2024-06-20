import numpy as np
from coinrun import setup_utils, make
import os
from PIL import Image


def random_agent(num_episodes=10000, max_steps=1000):
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/images", exist_ok=True)
    setup_utils.setup_and_load()

    for i in range(num_episodes):
        metadata = []
        print(f"Episode {i}")
        last_acts = None
        env = make('standard', num_envs=1)

        env.step(np.array([0]))
        obs = env.get_images()
        image = Image.fromarray(obs[0])
        image.save(f"dataset/images/episode_{i}_step_0.jpg")
        metadata.append({"obs": f"dataset/images/episode_{i}_step_0.jpg"})

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
            image.save(f"dataset/images/episode_{i}_step_{step+1}.jpg")

            dict_to_add = {
                "acts": acts[0].tolist(),
                "obs": f"images/episode_{i}_step_{step+1}.jpg",
                "rews": rews[0].tolist(),
                "dones": dones[0],
            }
            metadata.append(dict_to_add)

        env.close()

        with open(f"dataset/metadata.jsonl", "a") as f:
            f.write(f"{metadata}\n")


if __name__ == '__main__':
    random_agent()