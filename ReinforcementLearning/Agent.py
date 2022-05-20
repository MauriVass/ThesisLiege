import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Experiences import Experiences
from Networks import ActorNetwork, CriticNetwork


class Agent:
	def __init__(self, input_dims, alpha=0.001, beta=0.003, gamma=0.99,
				n_actions=2, scale_actions=1, max_size=1000000, tau=0.005,
				fc1=400, fc2=300, batch_size=64, noise=0.01):
		self.gamma = gamma
		self.tau = tau
		self.memory = Experiences(max_size, input_dims, n_actions)
		self.batch_size = batch_size
		self.n_actions = n_actions
		self.scale_actions = scale_actions
		self.noise = noise
		self.min_noise = 0.0001
		self.noise_reduc_factor = 0.9

		self.step_counter = 0
		self.step_copying = 1000
		self.step_learning = 2
		self.step_reducing_exploration = 2500

		self.actor = ActorNetwork(n_actions=n_actions, name='actor', fc1_dims=fc1, fc2_dims=fc2)
		self.critic = CriticNetwork(name='critic', fc1_dims=fc1, fc2_dims=fc2)
		self.target_actor = ActorNetwork(n_actions=n_actions,name='target_actor', fc1_dims=fc1, fc2_dims=fc2)
		self.target_critic = CriticNetwork(name='target_critic', fc1_dims=fc1, fc2_dims=fc2)

		self.actor.compile(optimizer=Adam(learning_rate=alpha))
		self.critic.compile(optimizer=Adam(learning_rate=beta))
		self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
		self.target_critic.compile(optimizer=Adam(learning_rate=beta))

		self.update_network_parameters(tau=1)

		self.history = []
		self.curtailment = []

	def update_network_parameters(self, tau=None):
		if tau is None:
			tau = self.tau

		weights = []
		targets = self.target_actor.weights
		for i, weight in enumerate(self.actor.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_actor.set_weights(weights)

		weights = []
		targets = self.target_critic.weights
		for i, weight in enumerate(self.critic.weights):
			weights.append(weight * tau + targets[i]*(1-tau))
		self.target_critic.set_weights(weights)

	def save_experience(self, state, action, reward, next_state, done):
		self.memory.save_experience(state, action, reward, next_state, done)

	def save_models(self):
		print('... saving models ...')
		self.actor.save_weights(self.actor.checkpoint_file)
		self.target_actor.save_weights(self.target_actor.checkpoint_file)
		self.critic.save_weights(self.critic.checkpoint_file)
		self.target_critic.save_weights(self.target_critic.checkpoint_file)

	def load_models(self):
		print('... loading models ...')
		self.actor.load_weights(self.actor.checkpoint_file)
		self.target_actor.load_weights(self.target_actor.checkpoint_file)
		self.critic.load_weights(self.critic.checkpoint_file)
		self.target_critic.load_weights(self.target_critic.checkpoint_file)

	def choose_action(self, observation, explore=False):
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		actions = self.actor(state)
		if explore:
			actions += tf.random.normal(shape=[self.n_actions],
										mean=0.0, stddev=self.noise)
			if(self.step_counter%self.step_reducing_exploration==0 and self.step_counter>self.step_reducing_exploration):
				self.noise = self.noise * self.noise_reduc_factor if self.noise>self.min_noise else self.min_noise
		# note that if the env has an action > 1, we have to multiply by
		# max action at some point
		actions = actions * self.scale_actions
		#Clip [0, actions) since curtailemnt can't be negative
		actions = tf.clip_by_value(actions, 0, actions)

		return actions[0]

	def learn(self):
		if self.memory.mem_counter < self.batch_size:
			return

		if(self.step_counter%self.step_learning==0):
			state, action, reward, new_state, done = \
				self.memory.sample_experiences(self.batch_size)

			states = tf.convert_to_tensor(state, dtype=tf.float32)
			new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
			rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
			actions = tf.convert_to_tensor(action, dtype=tf.float32)

			with tf.GradientTape() as tape:
				target_actions = self.target_actor(new_states)
				next_critic_value = tf.squeeze(self.target_critic(new_states, target_actions), 1)
				target = rewards + self.gamma*next_critic_value*(1-done)
				critic_value = tf.squeeze(self.critic(states, actions), 1)
				critic_loss = keras.losses.MSE(target, critic_value)

			critic_network_gradient = tape.gradient(critic_loss,self.critic.trainable_variables)
			self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

			with tf.GradientTape() as tape:
				new_policy_actions = self.actor(states)
				#Want to maximize this term so the '-' sign
				actor_loss = -self.critic(states, new_policy_actions)
				actor_loss = tf.math.reduce_mean(actor_loss)

			actor_network_gradient = tape.gradient(actor_loss,self.actor.trainable_variables)
			self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

			if(self.step_counter%self.step_copying==0):
				self.update_network_parameters()

	def increment_step_counter(self):
		self.step_counter += 1

	def save_agent(self, path):
		self.actor.save_weights(path)
		self.critic.save_weights(path)
	def load_agent(self, path):
		self.actor.load_weights(path)
		self.critic.load_weights(path)
