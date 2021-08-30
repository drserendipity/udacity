import numpy as np
import torch
import torch.nn.functional as F


class MultiAgent():
    def __init__(self, config, load_from_files=False):
        self.t_step = 0
        self.config = config

        if config.shared_replay_buffer:
            self.memory = config.memory_fn()
            self.config.memory = self.memory

        self.ddpg_agents = [DDPGAgent(self.config) for _ in range(config.num_agents)]
        if load_from_files:
            for i, ddpg_agent in enumerate(self.ddpg_agents):
                ddpg_agent.__init__(filename_actor="checkpoint_actor_" + str(i) + ".pth", filename_critic="checkpoint_critic_" + str(i) + ".pth")
        
        
    def step(self, state, action, reward, next_state, done, time_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.config.batch_size and time_step % self.config.learn_every == 0:
            for i in range(self.config.num_learn):
                for agent in self.ddpg_agents:
                    if self.config.shared_replay_buffer:
                        experiences = self.memory.sample()
                    else:
                        experiences = agent.memory.sample()

                    agent.learn(experiences, self.config.gamma)

#     def step(self, states, actions, rewards, next_states, dones):
#         # Save experience in replay memory
#         for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
#             self.memory.add(state, action, reward, next_state, done)

#         # Learn every UPDATE_EVERY time steps.
#         self.t_step = (self.t_step + 1) % self.config.learn_every
#         if self.t_step == 0:
#             # If enough samples are available in memory, get random subset and learn
#             if len(self.memory) > self.config.batch_size:
#                 for agent in self.ddpg_agents:
#                     if self.config.shared_replay_buffer:
#                         experiences = self.memory.sample()
#                     else:
#                         experiences = agent.memory.sample()

#                     agent.learn(experiences, self.config.gamma)                    
                    
    def act(self, all_states):
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions

    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, config, filename_actor=None, filename_critic=None):
        """Initialize an Agent object.
        """
        self.config = config
        self.seed = config.seed
        self.epsilon = self.config.epsilon

        # Actor Network (w/ Target Network)
        self.actor_local = config.actor_network_fn()
        self.actor_target = config.actor_network_fn()
        self.actor_optimizer = config.actor_optimizer_fn(self.actor_local.parameters())

        # Critic Network (w/ Target Network)
        self.critic_local = config.critic_network_fn()
        self.critic_target = config.critic_network_fn()
        self.critic_optimizer = config.critic_optimizer_fn(self.critic_local.parameters())

        if filename_actor and filename_critic:
            weights_actor = torch.load(filename_actor)
            self.actor_local.load_state_dict(weights_actor)
            self.actor_target.load_state_dict(weights_actor)

            weights_critic = torch.load(filename_critic)
            self.critic_local.load_state_dict(weights_critic)
            self.critic_target.load_state_dict(weights_critic)

        # Noise process
        self.noise = config.noise_fn()

        # Replay memory
        if config.shared_replay_buffer:
            self.memory = config.memory
        else:
            self.memory = config.memory_fn()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.config.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config.tau)
        self.soft_update(self.actor_local, self.actor_target, self.config.tau)

        # ----------------------- apply epsilon decay -------------------------- #
        self.epsilon -= self.config.epsilon_decay
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
