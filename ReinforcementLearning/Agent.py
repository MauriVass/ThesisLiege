import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from Experiences import Experiences
from Networks import ActorNetwork, CriticNetwork
import numpy as np


class Agent:
	def __init__(self, input_dims, alpha=0.001, beta=0.003, gamma=0.99,
				n_actions=2, scale_actions=1, max_size=1000000, tau=0.005,
				fc1=400, fc2=300, batch_size=32, noise=0.2, tensorboard_writer=None):
		self.gamma = gamma
		self.tau = tau
		self.memory = Experiences(max_size, input_dims, n_actions)
		self.batch_size = batch_size
		self.n_actions = n_actions
		self.scale_actions = scale_actions
		self.tensorboard_writer = tensorboard_writer

		self.step_counter = 0
		self.step_copying = 2000
		self.step_learning = 2
		self.step_reducing_exploration = 1000

		self.noise = noise
		self.min_noise = 0.0002
		self.noise_reduc_factor = 0.991

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
		self.losses = []
		self.gradients = []

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
		#NN expects a extra dimention for the batch so add one more with []
		state = tf.convert_to_tensor([observation], dtype=tf.float32)
		actions = self.actor(state)
		if explore:
			actions += tf.random.normal(shape=[self.n_actions],
										mean=0.0, stddev=self.noise)
			if(self.step_counter%self.step_reducing_exploration==0 and self.step_counter>self.step_reducing_exploration):
				self.noise = self.noise * self.noise_reduc_factor if self.noise>self.min_noise else self.min_noise
		
		#Clip [0, actions) since curtailemnt can't be negative
		actual_action_length = int(self.n_actions/2)

		p_actions = actions[0][:actual_action_length]
		p_actions = tf.clip_by_value(p_actions, clip_value_min=0, clip_value_max=1) #[0,1]
		
		q_actions = (actions[0][actual_action_length:])

		actions = tf.concat([p_actions, q_actions],0)
		#Reshape to get a single array
		actions = tf.reshape(actions,[1,self.n_actions])

		return actions[0] # the [0] is for the previously add extra batch dimention

	def learn(self):
		if (self.memory.mem_counter < self.batch_size):
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

			self.losses.append([critic_loss.numpy(),actor_loss.numpy()])
			# self.gradients.append([critic_network_gradient.numpy(),actor_network_gradient.numpy()])

			if(self.step_counter%100==0):
				with self.tensorboard_writer.as_default():
					tf.summary.scalar('critic_loss', critic_loss.numpy(), step=self.step_counter)
					tf.summary.scalar('actor_loss', actor_loss.numpy(), step=self.step_counter)
					for layer in self.actor.trainable_weights:
						tf.summary.histogram('actor_'+layer.name, layer.numpy(), step=self.step_counter)
					for layer in self.critic.trainable_weights:
						tf.summary.histogram('critic_'+layer.name, layer.numpy(), step=self.step_counter)

	def increment_step_counter(self):
		self.step_counter += 1