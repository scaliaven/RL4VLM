from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import sys; print(sys.path)

import copy
import glob
import os
import time
from collections import deque

import gymnasium as gym
import gym_cards
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils, rl_utils
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names, load_lora_model

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import math
import random
from functools import partial
from typing import List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoImageProcessor
import transformers

from tqdm import tqdm

import accelerate
from accelerate.state import AcceleratorState

import warnings
warnings.filterwarnings("ignore")

def text_process(obs, tokenizer):
    result = tokenizer_image_token(obs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    result[result == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
    return result

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    device = accelerator.device
    ## environment interaction device is cpu
    model_device = device

    #initialization of llava
    model_path = args.model_path
    cache_dir = args.cache_dir

    print(model_path)
    #load_pretrained_model(model_path, model_path, model_path)
    if "lora" in model_path:
        base, tokenizer = load_lora_model(model_path, cache_dir=cache_dir)
        if args.q8 or args.q4:
            raise ValueError("Lora model does not support 8bit or 4bit quantization")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        if args.q8:
            print("8bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, cache_dir=cache_dir)
        elif args.q4:
            q4_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                    )
            print("4bit quantization")
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, load_in_4bit=True, quantization_config=q4_config, cache_dir=cache_dir)
        else:
            if 'mistral' in model_path.lower():
                base =  LlavaMistralForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                base = LlavaLlamaForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    use_grad_ckpt = True
    if use_grad_ckpt:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    base.config.max_length = 1024
    print("Model max context length:{}".format(base.config.max_length))
    base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter = args.pretrain_mm_adapter)
    image_processor = base.get_vision_tower().image_processor
    # if input "image" is actually text
    if args.feature == "text":
        image_processor = tokenizer
        print(base)
        base.model.vision_tower = base.model.embed_tokens
        base.model.mm_projector = nn.Identity()
    elif args.feature == "tensor":
        image_processor = nn.Flatten()
        print(base)
        if "junqi" in args.env_name:
            in_features = 60
        elif "xiangqi" in args.env_name:
            in_features = 90
        else:
            raise Exception(f"Tensor feature not supported for {args.env_name}")
        base.model.vision_tower = nn.Linear(in_features, 4096)
        base.model.mm_projector = nn.Identity()

    base_lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=find_all_linear_names(base,args.train_vision),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if args.use_lora:
        base = get_peft_model(base, base_lora_config)
    value_model = VLMValue(base)
    value_model = value_model.to(model_device)

    if "gym_cards" in args.env_name.lower():
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.gamma, None, device, False, 1)
    elif "nlp" in args.env_name.lower():
        envs = make_vec_envs("Taxi-v3", args.seed, args.num_processes,
                             args.gamma, None, device, False)
    elif "xiangqi" in args.env_name.lower():
        envs = make_vec_envs("gym_xiangqi:xiangqi-v0", args.seed, args.num_processes,
                             args.gamma, None, device, False)
    elif "junqi" in args.env_name.lower():
        envs = make_vec_envs("gym_junqi:junqi-v0", args.seed, args.num_processes,
                             args.gamma, None, device, False)
    else:
        print("Environment not supported")
        exit(1)


    obs = envs.reset()
    if args.feature == "tensor":
        if "junqi" in args.env_name:
            UNKNOWN = 26
            obs[obs < 0] = -UNKNOWN
        obs = obs[0]
    if args.feature == "text":
        obs = list(envs.envs[0].decode(obs))
        obs = "taxi is in row {} and column {}. Passenger is located in location {} and want to move to location {}".format(obs[0].item(), obs[1].item(), obs[2].item(), obs[3].item())
        obs = text_process(obs, tokenizer)
    infos = None
    ## Inputing Prompt here
    qs = get_prompt(args.env_name, args.action_only_prompt, infos)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace

    projection_f = partial(text_projection, env_name=args.env_name)

    actor_critic = VLMPolicy(tokenizer=tokenizer,
                             image_processor=image_processor,
                             value_model=value_model,
                             projection_f=projection_f,
                             INPUT_IDS=INPUT_IDS,
                             args=args)
    optimizer = optim.Adam(actor_critic.value_model.parameters(), lr=args.init_lr, eps=args.eps, weight_decay=args.weight_decay)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_max_steps, eta_min=args.end_lr)

    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    actor_critic, optimizer, lr_scheduler = accelerator.prepare(actor_critic, optimizer, lr_scheduler)

    agent = algo.PPO(
            actor_critic,
            optimizer,
            accelerator,
            args.clip_param,
            args.ppo_epoch,
            args.mini_batch_size,
            args.value_loss_coef,
            args.entropy_coef,
            max_grad_norm=args.max_grad_norm)

    print(envs.observation_space.shape, obs.shape) # temporary fix
    if args.feature == "text":
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                obs.shape, envs.action_space, args.max_new_tokens)
    elif args.feature == "image":
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space, args.max_new_tokens)
    elif args.feature == "tensor":
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space, args.max_new_tokens)
    _, output_ids, action, action_log_prob, action_tokens_log_prob = actor_critic.act(obs, INPUT_IDS = INPUT_IDS, feature_type=args.feature)
    # print("Shape & Type of observation:", obs.shape, type(obs))
    print("action:{}".format(action))
    print("action_log_prob:{}".format(action_log_prob))
    print("action_tokens_log_prob:{}".format(action_tokens_log_prob))

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=args.eval_num_per_episode)
    episode_success_rate = deque(maxlen=args.eval_num_per_episode)
    episode_action_tokens_log_prob = deque(maxlen=args.eval_num_per_episode)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run + "-" + args.env_name
        wandb.init(project=args.wandb_project, name=run_name, group=run_name, config=args)

    print(qs)
    running_episode_rewards = torch.zeros(args.num_processes).flatten()

    num_explore = int(args.explore_portion*num_updates)
    prev_infos = []
    infos = []
    for j in range(num_updates):

        for step in tqdm(range(args.num_steps), desc=f'training for epoch {j}'):
            # Sample actions
            with torch.no_grad():
                INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                INPUT_IDS[INPUT_IDS == 0] = 259 # 869: . (period), 29871: SPIECE, 259: whitespace
                if args.feature == "text":
                    value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                            rollouts.obs[step][0], INPUT_IDS = INPUT_IDS, feature_type=args.feature)
                elif args.feature == "image":
                    value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                            rollouts.obs[step], INPUT_IDS = INPUT_IDS, feature_type=args.feature)
                elif args.feature == "tensor":
                    value, output_id, action, action_log_prob, action_tokens_log_prob = actor_critic.act(
                            rollouts.obs[step], INPUT_IDS = INPUT_IDS, feature_type=args.feature)
            text_action = tokenizer.decode(list(filter(lambda num: num != 0, output_id[0].tolist())))
            prev_infos = copy.deepcopy(infos)
            obs, reward, done, infos = envs.step(action)
            if args.feature == "text":
                obs = list(envs.envs[0].decode(obs))
                obs = "taxi is in row {} and column {}. Passenger is located in location {} and want to move to location {}".format(obs[0].item(), obs[1].item(), obs[2].item(), obs[3].item())
                obs = text_process(obs, tokenizer)
            elif args.feature == "tensor":
                if "junqi" in args.env_name:
                    UNKNOWN = 26
                    obs[obs < 0] = -UNKNOWN
                print(obs)
                obs = obs[0]

            qs = get_prompt(args.env_name, args.action_only_prompt, infos)
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            running_episode_rewards += reward.flatten()
            for i, d, r in zip(range(args.num_processes), done, reward):
                if d:
                    episode_rewards.append(running_episode_rewards[i].item())
                    if running_episode_rewards[i] > 0:
                        episode_success_rate.append(1)
                    else:
                        episode_success_rate.append(0)
                    episode_action_tokens_log_prob.append(action_tokens_log_prob[i].item())
                    running_episode_rewards[i] = 0
            # bad_mask is a legacy implementation of the storage.py file
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, output_id, action,
                            action_log_prob, value, reward, masks, bad_masks)
        print("****** iteration number:{} ******".format(j))
        print("prompt:{}".format(prompt))
        print("text_action:{}".format(text_action))
        print("current observation:{}".format(prev_infos))
        print("ground truth:{}".format(infos))
        print("action log prob:{}".format(action_log_prob))
        print("action tokens log prob:{}".format(action_tokens_log_prob))
        with torch.no_grad():
            if args.feature == "text":
                next_value = actor_critic.get_value(
                    rollouts.obs[-1][0], feature_type = args.feature).detach()
            elif args.feature == "image":
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], feature_type = args.feature).detach()
            elif args.feature == "tensor":
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], feature_type = args.feature).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts, args.feature)
        lr_scheduler.step()

        rollouts.after_update()
        if len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success_rate {:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(episode_success_rate),
                        dist_entropy, value_loss, action_loss))
            if args.use_wandb:
                wandb.log({"iteration": j,
                        "num_timesteps": total_num_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "episode_reward.mean": np.mean(episode_rewards),
                        "episode_reward.median": np.median(episode_rewards),
                        "episode_reward.min": np.min(episode_rewards),
                        "episode_reward.max": np.max(episode_rewards),
                        "episode_success_rate.mean": np.mean(episode_success_rate),
                        "episode_action_tokens_log_prob.mean": np.mean(episode_action_tokens_log_prob),
                        "distribution_entropy": dist_entropy,
                        "value.loss": value_loss,
                        "action.loss": action_loss,
                        "reward.max": rollouts.rewards.max().item(),
                        "reward.min": rollouts.rewards.min().item(),
                        "reward.mean": rollouts.rewards.mean().item(),
                        "reward.std": rollouts.rewards.std().item(),
                        "reward.median": rollouts.rewards.median().item(),
                        "return.max": rollouts.returns.max().item(),
                        "return.min": rollouts.returns.min().item(),
                        "return.mean": rollouts.returns.mean().item(),
                        "return.std": rollouts.returns.std().item(),
                        "value.max": rollouts.value_preds.max().item(),
                        "value.min": rollouts.value_preds.min().item(),
                        "value.mean": rollouts.value_preds.mean().item(),
                        "value.std": rollouts.value_preds.std().item(),})

if __name__ == "__main__":
    main()

