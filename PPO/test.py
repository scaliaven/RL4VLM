import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# Hyperparameters
learning_rate = 2.5e-4
gamma = 0.99
clip_eps = 0.2
update_epochs = 4
batch_size = 64
timesteps_per_batch = 2048

# use vec env
# def make_env():
#     return lambda: gym.make("Taxi-v3")

# Make env
env = gym.make("Taxi-v3")
obs_dim = 1  # Discrete obs
act_dim = env.action_space.n

# One-hot encoding of discrete states (n=500)
obs_n = env.observation_space.n

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(obs_n, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.policy = nn.Linear(64, act_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return self.policy(x), self.value(x)

model = ActorCritic()
print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Rollout buffer
class Buffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

buffer = Buffer()

def compute_returns(rewards, dones, values, gamma):
    returns = []
    R = 0
    for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
        R = r + gamma * R * (1 - d)
        returns.insert(0, R)
    return returns

# Training loop
def train(env, model, optimizer):
    
    obs, _ = env.reset()
    obs = torch.tensor([obs], dtype=torch.long)

    model = model.to(device)
    obs = obs.to(device)

    pbar = trange(500)
    for iteration in pbar:
        buffer.clear()
        for _ in range(timesteps_per_batch):
            logits, value = model(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buffer.obs.append(obs)
            buffer.actions.append(action)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value.squeeze())

            obs = torch.tensor([next_obs], dtype=torch.long).to(device)

            if done:
                obs, _ = env.reset()
                obs = torch.tensor([obs], dtype=torch.long).to(device)

        # Compute returns
        returns = compute_returns(buffer.rewards, buffer.dones, buffer.values, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        old_values = torch.stack(buffer.values)
        old_log_probs = torch.stack(buffer.log_probs).detach()
        observations = torch.cat(buffer.obs)
        actions = torch.stack(buffer.actions)

        # Normalize advantage
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(update_epochs):
            indices = np.arange(len(returns))
            np.random.shuffle(indices)
            for i in range(0, len(indices), batch_size):
                idx = indices[i:i + batch_size]
                batch_obs = observations[idx]
                batch_act = actions[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                batch_old_log = old_log_probs[idx]

                logits, value = model(batch_obs)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_act)

                ratio = torch.exp(log_probs - batch_old_log)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (value.squeeze() - batch_ret).pow(2).mean()

                loss = policy_loss + 0.25 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # logits / entropy debugging
                # if iteration % 25 == 0 and i == 0:
                #     print(f"[DEBUG] Iter {iteration}")
                #     print(f"  Logits: {logits[0].detach().cpu().numpy()}")
                #     print(f"  Probs: {torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()}")
                #     print(f"  Entropy: {dist.entropy()[0].item():.4f}")
                
        # value loss debugging    
        if iteration % 25 == 0:
            print(f"[DEBUG] Value loss: {value_loss.item():.2f}")

        avg_reward = np.mean(buffer.rewards)
        pbar.set_description(f"Iter {iteration} | Avg reward: {avg_reward:.2f} | Value loss: {value_loss:.2f} | Polic Loss: {policy_loss:.2f} | Loss: {loss:.2f}")
        # print(f"Iter {iteration}, avg reward: {np.mean(buffer.rewards):.2f}")
    
def success_fn(ep_reward, ep_len):
    return ep_len < 200    

def evaluate_policy(env, model, episodes=10, deterministic=True, success_fn=None):
    """
    Evaluate a policy over a number of episodes.

    Args:
        env: A gymnasium environment.
        model: ActorCritic model.
        episodes: Number of episodes to evaluate.
        deterministic: If True, use argmax(logits). Otherwise, sample actions.
        success_fn: A callable that takes (ep_reward, ep_len) and returns True if the episode is considered successful.

    Returns:
        A dict with avg_reward, success_rate, avg_episode_length, avg_entropy, and value_prediction_error.
    """
    model.eval()
    total_rewards = []
    success_count = 0
    episode_lengths = []
    entropies = []
    value_errors = []

    for _ in range(episodes):
        obs, info = env.reset()
        action_mask = None
        obs = torch.tensor([obs], dtype=torch.long).to(device)
        done = False
        ep_reward = 0
        ep_len = 0
        episode_values = []
        episode_returns = []

        success = False
        while not done:
            with torch.no_grad():
                logits, value = model(obs)
                if action_mask is not None:
                    mask = torch.tensor(action_mask,  # your custom mask: 1 legal, 0 illegal
                                        dtype=torch.bool, device=device)
                    illegal = ~mask                            # illegal positions = True
                    logits[0, illegal] = -1e9       
                dist = torch.distributions.Categorical(logits=logits)
                action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
                log_prob = dist.log_prob(action)
                entropies.append(dist.entropy().item())

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            action_mask = info["action_mask"]
            obs = torch.tensor([next_obs], dtype=torch.long)

            ep_reward += reward
            if reward == 20:
                success = True
            ep_len += 1
            episode_values.append(value.item())
            episode_returns.append(reward)

            done = terminated or truncated
            
            # print(f"[Eval] Predicted value: {value.item():.2f}, reward so far: {ep_reward}")

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

        # Count as success if success_fn says so, else use default heuristic (positive return)
        # if success_fn:
        #     success = success_fn(ep_reward, ep_len)
        # else:
        #     success = ep_reward >= 0
        success_count += int(success)

        # Value prediction error (VPE)
        returns = []
        R = 0
        for r in reversed(episode_returns):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        values = torch.tensor(episode_values, dtype=torch.float32)
        vpe = ((returns - values) ** 2).mean().item()
        value_errors.append(vpe)

    return {
        "avg_reward": np.mean(total_rewards),
        "success_rate": success_count / episodes,
        "avg_episode_length": np.mean(episode_lengths),
        "avg_entropy": np.mean(entropies),
        "value_prediction_error": np.mean(value_errors)
    }

import matplotlib.pyplot as plt

def plot_training_curves(iterations, avg_rewards, value_losses):
    """
    Plot average reward and value loss over training iterations, side by side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: Average Reward ---
    ax1.plot(iterations, avg_rewards, marker='o', linestyle='-')
    ax1.set_title("Iteration vs Average Reward", fontsize=14)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Avg Reward", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Value Loss ---
    ax2.plot(iterations, value_losses, marker='o', linestyle='-')
    ax2.set_title("Iteration vs Loss", fontsize=14)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    # plt.show()
    fig.savefig('plot.png')

train(env, model, optimizer)

eval_res = evaluate_policy(env, model, deterministic=False, success_fn=success_fn)
print(eval_res)

iterations   = [24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 
             274, 299, 324, 349, 374, 399, 424, 449, 474, 499]
avg_rewards  = [-3.75, -4.01, -3.47, -3.17, -2.25, -2.38, -2.01,
              -1.31, -1.08, -1.02, -1.00, -1.00, -1.00, -1.00,
              -1.00, -1.19, -1.00, -1.00, -1.38, -1.00]
losses       = [2549.53, 2124.57, 2126.25, 1738.66, 807.00, 727.05,
         749.75, 212.38, 169.50, 132.07, 136.77, 118.72,
         142.77, 162.68, 124.97, 275.58, 164.89, 153.17,
         217.63, 156.64]
plot_training_curves(iterations, avg_rewards, losses)

