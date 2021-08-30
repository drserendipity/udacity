
class Config:
    def __init__(self):
        self.device = 'cpu'
        self.seed = 2
        self.num_agents = 1
        self.states = None
        self.state_size = None
        self.action_size = None
        self.activation = None

        self.network_fn = None
        self.optimizer_fn = None
        self.noise_fn = None
        self.hidden_units = None

        self.shared_replay_buffer = True
        self.memory = None
        self.memory_fn = None

        self.actor_hidden_units = (512, 256)
        self.actor_network_fn = None
        self.actor_optimizer_fn = None
#         self.actor_learning_rate = 2e-4
        self.actor_learning_rate = 1e-4
        self.actor_weight_decay = 0

        self.critic_hidden_units = (512, 256)
        self.critic_network_fn = None
        self.critic_optimizer_fn = None
#         self.critic_learning_rate = 2e-4
        self.critic_learning_rate = 3e-4
        self.critic_weight_decay = 0
        
        self.buffer_size = int(1e6)
        # self.buffer_size = int(1e5)
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 1e-3        

        self.epsilon = 1.0  # initial value of epsilon
        self.epsilon_decay = 1e-6  # decay value for epsilon (epsilon -> epsilon - epsilon_decay)
#         self.learn_every = 5  # time-step interval for learning
        self.learn_every = 10  # time-step interval for learning
        self.num_learn = 5  # number of learning with sampleing memory