import numpy as np
import matplotlib.pyplot as plt
import os

def plot_episode_rewards(save_dir, episode_rewards, optimal=None, label="MARL", filename="training_curve.png"):
    x = np.arange(1, len(episode_rewards) + 1)
    plt.figure(figsize=(6,4))
    plt.plot(x, episode_rewards, color="purple", linewidth=2, label=label)
    if optimal is not None:
        plt.axhline(y=optimal, color="black", linestyle="--", linewidth=1, label="Optimal")
    plt.xlabel("Episodes(25 time-step)")
    plt.ylabel("Episode Reward")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close()

