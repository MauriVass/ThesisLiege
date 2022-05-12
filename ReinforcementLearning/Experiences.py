import numpy as np

class Experiences:
	def __init__(self, max_size, input_shape, n_actions):
		self.mem_size = max_size
		self.mem_counter = 0
		self.state_memory = np.zeros((self.mem_size, input_shape))
		self.action_memory = np.zeros((self.mem_size, n_actions))
		self.reward_memory = np.zeros(self.mem_size)
		self.next_state_memory = np.zeros((self.mem_size, input_shape))
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def save_experience(self, state, action, reward, next_state, done):
		index = self.mem_counter % self.mem_size

		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.terminal_memory[index] = done

		self.mem_counter = self.mem_counter+1

	def sample_experiences(self, batch_size):
		#Set a upper limit since you do not want to sample dummy experiences with 0s as values
		max_mem = np.min([self.mem_counter, self.mem_size])

		batch = np.random.choice(max_mem, batch_size, replace=False)

		states = self.state_memory[batch]
		states_ = self.next_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]

		return states, actions, rewards, states_, dones

	def store_experiences_on_file(self):
		#TBD
		return
	def load_experiences_from_file(self):
		#TBD
		return