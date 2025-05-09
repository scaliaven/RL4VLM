import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.llava_interface import llava_evaluate, llava_generate
import torch.nn.init as init

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class VLMValue(nn.Module):
    """
    actually the base is also used for generation!
    """
    def __init__(self, base):
        super(VLMValue, self).__init__()
        self.base = base
        # hard-code value head
        # self.value_head = nn.Linear(4096, 1, bias=True).to(base.device, dtype=torch.float16)
        self.value_head = nn.Sequential(
            nn.Linear(4096, 1024), # First layer
            nn.ReLU(), # Non-linearity
            nn.Linear(1024, 512), # Second layer
            nn.ReLU(), # Non-linearity
            nn.Linear(512, 1) # Output layer
            ).to(base.device, dtype=torch.float16) # Move to specified device with dtype

    def forward(self, input_ids, image_tensor, feature_type="image"):
        if image_tensor.size(0) != 1:
            input_ids = input_ids.broadcast_to(image_tensor.size(0), input_ids.size(-1))
        if feature_type == "image":
            image_tensor = image_tensor.to(self.base.device, dtype = self.base.dtype)
        elif feature_type == "tensor":
            image_tensor = image_tensor.to(self.base.device, dtype = self.base.dtype)
        elif feature_type == "text":
            image_tensor = image_tensor.to(self.base.device, dtype = torch.int64)
        _, _, _, _, inputs_embeds, _ = self.base.prepare_inputs_labels_for_multimodal(input_ids.to(self.base.device), None, None, None, None, image_tensor)
        inputs_embeds = inputs_embeds.to(self.base.device, dtype = self.base.dtype)
        if feature_type == "image":
            assert inputs_embeds.shape[1] > 256 #unsure
        outputs = self.base(
            inputs_embeds = inputs_embeds,
            output_hidden_states=True)
        hidden_states = outputs.hidden_states
        values = self.value_head(hidden_states[-1][:, -1])
        return values


class VLMPolicy(nn.Module):
    def __init__(self, tokenizer,
                image_processor,
                value_model,
                args,
                INPUT_IDS,
                projection_f,
                base_kwargs=None):
        """
        projection_f: the postprocessing function to parse text action
        """
        super(VLMPolicy, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.value_model = value_model
        self.base = value_model.base
        self.INPUT_IDS = INPUT_IDS
        self.projection_f = projection_f

    def process_obs(self, obs, feature_type="image"):
        #process the observation with the image processor
        if feature_type == "image":
            result = self.image_processor.preprocess(obs, return_tensors='pt')['pixel_values'].to(dtype=self.base.dtype)
        elif feature_type == "text":
            result = obs
        elif feature_type == "tensor":
            result = obs.reshape(1, 6, -1)

        return result

    def act(self, inputs, deterministic=False, INPUT_IDS=None, feature_type="image"):
        # print(type(inputs), type(INPUT_IDS))
        image_tensor = self.process_obs(inputs, feature_type)
        # print(type(image_tensor), image_tensor.shape)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, output_ids, text_action, action_log_prob, action_tokens_log_prob = llava_generate(value_model = self.value_model,
                                                    tokenizer = self.tokenizer,
                                                    input_ids = INPUT_IDS,
                                                    image_tensor = image_tensor,
                                                    args = self.args)
        action = self.projection_f(text_action)
        return value, output_ids, action, action_log_prob, action_tokens_log_prob

    def get_value(self, inputs, INPUT_IDS=None, feature_type="image"):
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        image_tensor = self.process_obs(inputs, feature_type)
        return self.value_model(input_ids = INPUT_IDS, image_tensor = image_tensor, feature_type = feature_type)

    def evaluate_actions(self, inputs, output_ids, INPUT_IDS=None, feature_type="image"):
        image_tensor = self.process_obs(inputs, feature_type)
        if INPUT_IDS is None:
            INPUT_IDS = self.INPUT_IDS
        value, action_log_prob, _ = llava_evaluate(value_model = self.value_model,
                                        input_ids = INPUT_IDS,
                                        output_ids = output_ids,
                                        image_tensor = image_tensor,
                                        temperature = self.args.temperature,
                                        thought_prob_coef = self.args.thought_prob_coef,
                                        feature_type = feature_type)
        return value, action_log_prob
