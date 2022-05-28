import os
import tensorflow as tf

class CriticNetwork(tf.keras.Model):
	def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
		super(CriticNetwork, self).__init__()
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.alpha = 0.1

		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')

		self.fc1 = tf.keras.layers.Dense(self.fc1_dims, activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
		# self.bn1 = tf.keras.layers.BatchNormalization()
		self.fc2 = tf.keras.layers.Dense(self.fc2_dims, activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
		# self.bn2 = tf.keras.layers.BatchNormalization()
		self.fc3 = tf.keras.layers.Dense(self.fc2_dims, activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
		# self.bn3 = tf.keras.layers.BatchNormalization()
		self.q = tf.keras.layers.Dense(1, activation=None)

	def call(self, state, action):
		action_value = self.fc1(tf.concat([state, action], axis=1))
		# action_value = self.bn1(action_value)
		action_value = self.fc2(action_value)
		# action_value = self.bn2(action_value)
		action_value = self.fc3(action_value)
		# action_value = self.bn3(action_value)

		q = self.q(action_value)

		return q

class ActorNetwork(tf.keras.Model):
	def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor', chkpt_dir='tmp/ddpg'):
		super(ActorNetwork, self).__init__()
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.alpha = 0.1

		self.model_name = name
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')

		self.fc1 = tf.keras.layers.Dense(self.fc1_dims, activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
		# self.bn1 = tf.keras.layers.BatchNormalization()
		self.fc2 = tf.keras.layers.Dense(self.fc2_dims, activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
		# self.bn2 = tf.keras.layers.BatchNormalization()
		self.fc3 = tf.keras.layers.Dense(self.fc2_dims, activation=tf.keras.layers.LeakyReLU(alpha=self.alpha))
		# self.bn3 = tf.keras.layers.BatchNormalization()
		self.mu = tf.keras.layers.Dense(int(self.n_actions/2), activation='sigmoid')
		self.muq = tf.keras.layers.Dense(int(self.n_actions/2), activation='tanh')
		# self.muq = tf.keras.layers.Dense(int(self.n_actions), activation='tanh')

	def call(self, state):
		prob = self.fc1(state)
		# prob = self.bn1(prob)
		prob = self.fc2(prob)
		# prob = self.bn2(prob)
		prob = self.fc3(prob)
		# prob = self.bn3(prob)

		mu = self.mu(prob)
		muq = self.muq(prob)

		p_actions = mu
		
		q_actions = (muq * 0.5) #[-0.5,0.5]
		mu = tf.concat([p_actions,q_actions],0)
		mu = tf.reshape(mu,[state.shape[0],self.n_actions])
		return mu