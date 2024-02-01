import numpy as np
import torch
import torch.nn as nn
import loralib as lora
import transformers
from .gpt2 import GPT2Model

device = torch.device("cuda")

class InnerConfig:
    def __init__(self):
        self.patch_dim = 11
        self.num_blocks = 1
        self.embed_dim_inner = 128
        self.num_heads_inner = 1
        self.attention_dropout_inner = 0.0
        self.ffn_dropout_inner = 0.0
        self.activation_fn_inner = nn.ReLU
        self.dim_expand_inner = 1
        self.have_position_encoding = False
        self.share_tit_blocks = False


class InnerLSTMBlock(nn.Module):
    def __init__(self, config):
        super(InnerLSTMBlock, self).__init__()
        self.ln = nn.LayerNorm(config.embed_dim_inner)
        self.input_size = config.embed_dim_inner
        self.hidden_size = config.embed_dim_inner
        self.num_layers = 1
        self.output_size = config.embed_dim_inner
        self.num_directions = 2  # 双向LSTM
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                                  bidirectional=True, batch_first=True)
        self.lin = torch.nn.Linear(2 * self.hidden_size, self.output_size)
        # self.lin = lora.Linear(2 * self.hidden_size, self.output_size, r=16)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        x_ln = self.ln(x)
        output, _ = self.lstm(x_ln, (h_0.detach(), c_0.detach()))
        pred = self.lin(output)
        return pred


class PLDT(nn.Module):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            obs_upper_bound,
            obs_lower_bound,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)
        self.prompt_embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.prompt_embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        inner_config = InnerConfig()
        inner_config.patch_dim = state_dim

        inner_config.embed_dim_inner = self.hidden_size
        print('inner_config.patch_dim ==>', inner_config.patch_dim, state_dim, self.hidden_size)
        self.inner_blocks = nn.ModuleList([InnerLSTMBlock(inner_config) for _ in range(inner_config.num_blocks)])
        self.obs_patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=inner_config.embed_dim_inner,
            kernel_size=inner_config.patch_dim,
            stride=inner_config.patch_dim,
            bias=False,
        )
        self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, inner_config.embed_dim_inner))
        nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        self.noise_sigma = 3e-3
        self.obs_upper_bound = obs_upper_bound
        self.obs_lower_bound = obs_lower_bound

    def _observation_patch_embedding(self, obs):
        B, context_len_outer, D = obs.size()
        B = B * context_len_outer
        obs = obs.view(B, D)
        obs = torch.unsqueeze(obs, dim=1)
        obs_patch_embedding = self.obs_patch_embed(obs)
        obs_patch_embedding = obs_patch_embedding.transpose(2, 1)
        return obs_patch_embedding

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        patch_embeddings = self._observation_patch_embedding(states)
        context_len_inner = patch_embeddings.shape[1]
        inner_tokens = torch.cat([self.class_token_encoding.expand(batch_size * seq_length, -1, -1), patch_embeddings],
                                 dim=1)
        for inner_block in self.inner_blocks:
            inner_tokens = inner_block(inner_tokens)
        temp = inner_tokens.view(batch_size, seq_length, context_len_inner + 1, self.hidden_size)
        state_embeddings = temp[:, :, 0, :]

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]
            # prompt_state_embeddings = self.prompt_embed_state(prompt_states)
            prompt_action_embeddings = self.prompt_embed_action(prompt_actions)
            if prompt_returns_to_go.shape[1] % 10 == 1:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps)

            prompt_patch_embeddings = self._observation_patch_embedding(prompt_states)
            # 1280=64*20
            prompt_context_len_inner = prompt_patch_embeddings.shape[1]
            inner_tokens = torch.cat(
                [self.class_token_encoding.expand(prompt_states.shape[0] * prompt_states.shape[1], -1, -1), prompt_patch_embeddings],
                dim=1)
            # torch.Size([1280, 12, 128])
            for inner_block in self.inner_blocks:
                inner_tokens = inner_block(inner_tokens)
            prompt_temp = inner_tokens.view(prompt_states.shape[0], prompt_states.shape[1], prompt_context_len_inner + 1, self.hidden_size)
            # torch.Size([64, 20, 12, 128])
            prompt_state_embeddings = prompt_temp[:, :, 0, :]
            # torch.Size([64, 20, 128])

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

            if prompt_stacked_inputs.shape[1] == 3 * seq_length:
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            else:
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        if prompt is None:
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]
        state_preds = self.predict_state(x[:,2])[:, -seq_length:, :]
        action_preds = self.predict_action(x[:,1])[:, -seq_length:, :]

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
