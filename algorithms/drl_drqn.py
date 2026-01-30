"""
This file includes the algorithm structure that is used for resource allocation.
Features:
    Adding LSTM layer as an input of the DQN network.
    Adding reward or agent index as a part of the state which can be adjusted from config file.
    Training with different batch size and number are part of the config file.
"""
import tensorflow as tf
import sys, os
import numpy as np
from .policies import BoltzmanPolicy, SoftmaxPolicy, EpsilonGreedy, GreedyPolicy, RandomPolicy
from collections import deque
#tf.config.optimizer.set_jit(True) # Enable XLA.
#tf.debugging.set_log_device_placement(True)

class DRQN:
    """
    Deep Recurrent Q Learning algorithm for distributed and dynamic resource allocation problem.
    This class can be used both for realness and test simulator.
    """
    def __init__(self, env, name="DeepRQN", total_episodes=4000, **kwargs):
        """
        Initialize the required variables and also the network.
        :param env: Either realness or test environment.
        :param name: Name of the scenario to test
        :param kwargs: Additional parameters.
        """
        tf.reset_default_graph()
        self.name = name  # Name of the experiment
        self.num_users = env.get_total_users()  # Number of users in the test scenario
        #========================== Add parameters of agent =============================#
        self.learning_rate = kwargs.setdefault("learning_rate", 1e-4)  # Learning rate of the DQN algorithm.
        self.target_update = kwargs.setdefault("target_update", 10)   # every determined time step, we update the target values with the current DQN network parameters(stabilizes learning)
        self.batch_size = kwargs.setdefault("batch_size", 64)   # Train the network with this determined batch_size, changing this value also affect the learning
        self.step_size = kwargs.setdefault("step_size", 5)      # This variable is used for LSTM network, observe not only the last one t but also within the interval [t-step_size, t] as a part of the state.
        self.n_batch = kwargs.setdefault("n_batch", 2)          # Number of times we train the our Q network.
        self.hysteretic = kwargs.setdefault("hysteretic", False) # This is a special technique that I wanted to test. Decreasing the learning rate when TD error is lowe can stabilizes the performance.
        self.state_size = env.get_state_space()                 # How big the state depends on the num users and and add reward and index.
        self.action_size = env.get_action_space()               # Number of available channels that UE can select defines the action space
        self.beta = kwargs.setdefault("beta", 1)                # variable is used for boltzman training
        self.explore_start = kwargs.setdefault("explore_start", 4)  # variable is used for boltzman training
        self.explore_stop = kwargs.setdefault("explore_stop", 4)  # variable is used for boltzman training
        self.decay_rate = kwargs.setdefault("decay_rate", 4)  # variable is used for boltzman training
        self.alpha = kwargs.setdefault("alpha", 1)  # variable is used for eps greedy policy
        self.eps = kwargs.setdefault("eps_init", 1)  # variable is used for eps greedy policy
        self.eps_decay = kwargs.setdefault("eps_decay", 0.9999)
        self.temperature = kwargs.setdefault("temperature", 0.001)

        self.network_param = kwargs.setdefault("network", False)
        self.use_lstm_input = self.network_param["use_lstm_input"]  # Use the lstm layer as an input to the DQN
        self.use_dueling = self.network_param["use_dueling"]        # An approach to improve the performance
        self.use_double = self.network_param["use_double"]          # Another approach to improve the performance
        self.nn_layers = self.network_param["layers"]               # Num of deep layers in the network.
        if 1: #with tf.device("cpu"):
        # ========================== Build Network ============================= #
            if self.use_lstm_input:
                self.inputs_ = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_')
            else:
                self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            self.one_hot_actions = tf.one_hot(self.actions_, self.action_size)
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

        # Create the network and training parameters below.
            with tf.variable_scope(name):
                with tf.variable_scope("eval_net_scope"):
                    self.eval_scope_name = tf.get_variable_scope().name
                    self.qvalues = self._create_network()

                with tf.variable_scope("target_net_scope"):
                    self.target_scope_name = tf.get_variable_scope().name
                    self.target_qvalues = self._create_network()

            # ============== LOSS ============== #
                self.gamma = kwargs.setdefault("gamma", 0.99)
                self.Q = tf.reduce_sum(tf.multiply(self.qvalues, self.one_hot_actions), axis=1)
                self.h_loss = self.Q - self.targetQs_  # Loss value is calculated by Hysteretic theorem.
                if self.hysteretic:
                    self.h_loss = tf.where(self.h_loss<0, self.h_loss/10, self.h_loss)
                self.loss = tf.reduce_mean(tf.square(self.h_loss))
                self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.h_loss)))
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.eval_scope_name)
                gradients = tf.gradients(self.loss, train_vars)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                grads_and_vars = [(g, v) for g, v in zip(clipped_gradients, train_vars) if g is not None]
                self.opt = optimizer.apply_gradients(grads_and_vars)

        # target network update op
            self.update_target_op = []
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
            for i in range(len(t_params)):
                self.update_target_op.append(tf.assign(t_params[i], e_params[i]))

        # init tensorflow session
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

        #################### Create Policy ######################
            self.policy_name = kwargs.setdefault("policy", "")
            if self.policy_name == "softmax":
                self.policy = SoftmaxPolicy(nA=self.action_size, temperature=self.temperature, episodes=total_episodes)
            elif self.policy_name == "boltzman":
                self.policy = BoltzmanPolicy(self.action_size, beta=self.beta, explore_start=self.explore_start,
                                        explore_stop=self.explore_stop, decay_rate=self.decay_rate, alpha=self.alpha)
            elif self.policy_name == "eps_greedy":
                self.policy = EpsilonGreedy(eps_init=self.eps, eps_decay=self.eps_decay, nA=self.action_size, episodes=total_episodes, explore_stop=self.explore_stop)
            else:
                self.policy = GreedyPolicy(nA=self.action_size)


    def _create_network(self):
        """
        Creates a deep neural network as a function approximation of DQN algorithm.
        :return: Model for networks
        """
        hidden_size = list(self.nn_layers.values())
        #hidden_size = hidden_size[0]
        if self.use_lstm_input:
            lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size[0])
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=0.9, output_keep_prob=0.9, state_keep_prob=0.9)
            #lstm = tf.contrib.rnn.GRUCell(hidden_size)
            lstm_out, state = tf.nn.dynamic_rnn(lstm, self.inputs_, dtype=tf.float32)
            reduced_out = lstm_out[:,-1,:]
            reduced_out = tf.reshape(reduced_out, shape=[-1, hidden_size[0]])

        else:
            w1 = tf.Variable(tf.random_uniform([self.state_size, hidden_size[0]]))
            b1 = tf.Variable(tf.constant(0.1, shape=[hidden_size[0]]))
            h1 = tf.matmul(self.inputs_, w1) + b1
            h1 = tf.nn.relu(h1)
            h1 = tf.contrib.layers.layer_norm(h1)

        w2 = tf.Variable(tf.random_uniform([hidden_size[0], hidden_size[1]]))
        b2 = tf.Variable(tf.constant(0.1, shape=[hidden_size[1]]))
        if self.use_lstm_input:
            h2 = tf.matmul(reduced_out, w2) + b2
        else:
            h2 = tf.matmul(h1, w2) + b2
        h2 = tf.nn.relu(h2)
        h2 = tf.contrib.layers.layer_norm(h2)

        if len(hidden_size) == 3:
            w3 = tf.Variable(tf.random_uniform([hidden_size[1], hidden_size[2]]))
            b3 = tf.Variable(tf.constant(0.1, shape=[hidden_size[2]]))
            h3 = tf.matmul(h2, w3) + b3
            h3 = tf.nn.relu(h3)
            h3 = tf.contrib.layers.layer_norm(h3)

            w4 = tf.Variable(tf.random_uniform([hidden_size[2], self.action_size]))
            b4 = tf.Variable(tf.constant(0, 1, shape=[self.action_size]))
            output = tf.matmul(h3, w4) + b4

        elif len(hidden_size) == 2:
            w3 = tf.Variable(tf.random_uniform([hidden_size[1], self.action_size]))
            b3 = tf.Variable(tf.constant(0, 1, shape=[self.action_size]))
            output = tf.matmul(h2, w3) + b3

        return output

    def infer_action(self, user, state_vector, episode, policy="boltzman"):
        """
        This function is used to take the action based on input.
        :param user: Id os the user
        :param state_vector: State of the user
        :param time_slot: Given time that the UE takes the decision.
        :param policy: Which policy to be applied for learning
        :return: action determined by the policy, a
        """
        #feeding the input-history-sequence of (t-1) slot for each user seperately
        if policy == "explore":
            action = np.random.randint(self.action_size)
            return action

        if self.use_lstm_input:
            feed = {self.inputs_:state_vector[:, user].reshape(1, self.step_size, self.state_size)}
        else:
            feed = {self.inputs_:state_vector[-1:,user].reshape(1, self.state_size)}
        Qs = self.sess.run(self.qvalues, feed_dict=feed)

        if policy =="greedy":
            action = np.argmax(Qs, axis=1)
        else:
            action = self.policy.action(Qs, episode)

        return action

    def set_eps(self, eps):
        """
        Used when we load the model, then we can start from a specific eps.
        :param eps:
        :return:
        """
        self.policy.set_epsilon(eps)

    def get_eps(self):
        """
        Get the eps value of the model
        :return: self.eps
        """
        return self.policy.get_epsilon()

    def train(self, sample_buffer, time_step):
        """
        Train the model based on the collected experience. Update the target network time to time.
        :param sample_buffer:
        :param time_step:
        :return:
        """
        # TODO: add training freq.
        n_batches = self.n_batch
        for k in range(n_batches):

            #  sampling a batch from memory buffer for training
            if self.use_lstm_input:
                batch = sample_buffer.sample(self.batch_size, self.step_size)
            else:
                batch = sample_buffer.sample(self.batch_size, 1)

            #   matrix of rank 4
            #   shape [NUM_USERS,batch_size,step_size,state_size]
            states = self.get_states_user(batch)

            #   matrix of rank 3
            #   shape [NUM_USERS,batch_size,step_size]
            actions = self.get_actions_user(batch)

            #   matrix of rank 3
            #   shape [NUM_USERS,batch_size,step_size]
            rewards = self.get_rewards_user(batch)

            #   matrix of rank 4
            #   shape [NUM_USERS,batch_size,step_size,state_size]
            next_states = self.get_next_states_user(batch)

            #   Converting [NUM_USERS,batch_size]  ->   [NUM_USERS * batch_size]
            #   first two axis are converted into first axis
            if self.use_lstm_input:
                states = np.reshape(states, [-1, states.shape[2], states.shape[3]])
                actions = np.reshape(actions, [-1, actions.shape[2]])
                rewards = np.reshape(rewards, [-1, rewards.shape[2]])
                next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])
            else:
                states = np.reshape(states, [-1, states.shape[3]])
                actions = np.reshape(actions, [-1])
                rewards = np.reshape(rewards, [-1])
                next_states = np.reshape(next_states, [-1, next_states.shape[3]])

            # creating target vector (possible best action)
            # target_Qs = self.sess.run(self.target_qvalues, feed_dict={self.inputs_:next_states})
            #  creating target vector (possible best action)
            #target_Qs = self.sess.run(self.qvalues, feed_dict={self.inputs_: next_states})

            #  Q_target =  reward + gamma * Q_next
            # targets = rewards[:,-1] + self.gamma * np.max(target_Qs, axis=1)
            targets = self._calc_target(rewards, next_states)
            #  calculating loss and train using Adam  optimizer
            if self.use_lstm_input:
                acts = actions[:,-1]
            else:
                acts = actions
            loss, _ = self.sess.run([self.loss, self.opt],
                                        feed_dict={self.inputs_:states,
                                        self.targetQs_:targets,
                                        self.actions_:acts})

        if (time_step+1) % self.target_update == 0:
            # print("Target Q update ct " + str(time_step))
            self.sess.run(self.update_target_op)

    def _calc_target(self, rewards, next_states):
        """
        Calculates the TD(temporal difference) target to be used for training.
        :param rewards: matrix with size (num_users*batch_size, step size)
        :param next_states: matrix with size (num_users*batch_size, step_size, state_space)
        :return:
        """
        n = len(rewards)
        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues],
                                               feed_dict={self.inputs_: next_states})
            act = np.argmax(qvalues, axis=1)

            next_value = t_qvalues[np.arange(n), act]

        else:
            t_qvalues = self.sess.run(self.target_qvalues, feed_dict={self.inputs_: next_states})
            next_value = np.max(t_qvalues, axis=1)

        #  Q_target =  reward + gamma * Q_next
        if self.use_lstm_input:
            target = rewards[:, -1] + self.gamma * next_value
        else:
            target = rewards + self.gamma * next_value

        return target

    def get_states_user(self, batch):
        """
        Receives a batch and returns the collected states of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return:
        """
        states = []
        for user in range(self.num_users):
            states_per_user = []
            for each in batch:
                states_per_batch = []
                for step_i in each:
                    try:
                        states_per_step = step_i[0][user]

                    except IndexError:
                        print (step_i)
                        print ("-----------")

                        print ("error")

                        '''for i in batch:
                            print i
                            print "**********"'''
                        sys.exit()
                    states_per_batch.append(states_per_step)
                states_per_user.append(states_per_batch)
            states.append(states_per_user)
        #print len(states)
        return np.array(states)

    def get_actions_user(self, batch):
        """
        Receives a batch and returns the collected actions of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return: collected actions
        """
        actions = []
        for user in range(self.num_users):
            actions_per_user = []
            for each in batch:
                actions_per_batch = []
                for step_i in each:
                    actions_per_step = step_i[1][user]
                    actions_per_batch.append(actions_per_step)
                actions_per_user.append(actions_per_batch)
            actions.append(actions_per_user)
        return np.array(actions)

    def get_rewards_user(self, batch):
        """
        Receives a batch and returns the collected rewards of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return: collected rewards
        """
        rewards = []
        for user in range(self.num_users):
            rewards_per_user = []
            for each in batch:
                rewards_per_batch = []
                for step_i in each:
                    rewards_per_step = step_i[2][user]
                    rewards_per_batch.append(rewards_per_step)
                rewards_per_user.append(rewards_per_batch)
            rewards.append(rewards_per_user)
        return np.array(rewards)

    def get_next_states_user(self, batch):
        """
        Receives a batch and returns the the next states of each user
        :param batch: (batch_size, step_size, (states_step, actions_step, rewards_step, next_state_step))
        :return: collected next states.
        """
        next_states = []
        for user in range(self.num_users):
            next_states_per_user = []
            for each in batch:
                next_states_per_batch = []
                for step_i in each:
                    next_states_per_step = step_i[3][user]
                    next_states_per_batch.append(next_states_per_step)
                next_states_per_user.append(next_states_per_batch)
            next_states.append(next_states_per_user)
        return np.array(next_states)

    def sample(self):
        """
        Sample action randomly to be used for initilization phase.
        """
        action_sampled = np.random.randint(self.action_size)
        return action_sampled

    def save_model(self, dir_name, slot, simulation):
        """
        Save model to dir
        :param dir_name: str
            Name of the directory
        :param epoch: int
        :return:
        """
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(dir_name, self.name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        saver.save(self.sess, os.path.join(dir_name, ("sim_%d_%d") % (simulation, slot)))

    def load_model(self, dir_name, epoch=0, name=None):
        """
        load model from dir
        :param dir_name: str
            name of the directory
        :param epoch:
        :param name:
        :return:
        """
        if name is None or name == self.name:  # the name of saved model is the same as ours
            dir_name = os.path.join(dir_name, self.name)
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            #model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            saver = tf.train.Saver(model_vars)
            # dir_name = dir_name + "/sim_0"
            saver.restore(self.sess, os.path.join(dir_name, ("sim_0_%d") % epoch))
        else:  # load a checkpoint with different name
            backup_graph = tf.get_default_graph()
            kv_dict = {}

            # load checkpoint from another saved graph
            with tf.Graph().as_default(), tf.Session() as sess:
                tf.train.import_meta_graph(os.path.join(dir_name, name, (self.subclass_name + "_%d") % epoch + ".meta"))
                dir_name = os.path.join(dir_name, name)
                model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(model_vars)
                saver.restore(sess, os.path.join(dir_name, (self.subclass_name + "_%d") % epoch))
                for item in tf.global_variables():
                    kv_dict[item.name] = sess.run(item)

            # assign to now graph
            backup_graph.as_default()
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            for item in model_vars:
                old_name = item.name.replace(self.name, name)
                self.sess.run(tf.assign(item, kv_dict[old_name]))
"""
This file includes the algorithm structure that is used for resource allocation.

UPDATED:
- Replaced LSTM (DRQN) with Transformer Encoder (Multi-Head Attention) for sequential inputs.
- Added neighbor-attention (within 200m) to focus on key neighbors dynamically.
- Kept same main-code interface: infer_action(user, state_vector, episode, policy)
"""

import tensorflow as tf
import sys, os
import numpy as np

from .policies import BoltzmanPolicy, SoftmaxPolicy, EpsilonGreedy, GreedyPolicy, RandomPolicy


class DRQN_TRANSFORMER:
    """
    Transformer-based DRQN (keeps same external API as original DRQN).
    """

    def _init_(self, env, name="DeepRQN", total_episodes=4000, **kwargs):
        tf.reset_default_graph()

        self.name = name
        self.num_users = env.get_total_users()

        # ========================== Add parameters of agent =============================#
        self.learning_rate = kwargs.setdefault("learning_rate", 1e4)
        self.target_update = kwargs.setdefault("target_update", 10)
        self.batch_size = kwargs.setdefault("batch_size", 64)
        self.step_size = kwargs.setdefault("step_size", 5)
        self.n_batch = kwargs.setdefault("n_batch", 2)
        self.hysteretic = kwargs.setdefault("hysteretic", False)

        self.state_size = env.get_state_space()
        self.action_size = env.get_action_space()

        self.beta = kwargs.setdefault("beta", 1)
        self.explore_start = kwargs.setdefault("explore_start", 4)
        self.explore_stop = kwargs.setdefault("explore_stop", 4)
        self.decay_rate = kwargs.setdefault("decay_rate", 4)

        self.alpha = kwargs.setdefault("alpha", 1)
        self.eps = kwargs.setdefault("eps_init", 1)
        self.eps_decay = kwargs.setdefault("eps_decay", 0.9999)
        self.temperature = kwargs.setdefault("temperature", 0.001)

        self.network_param = kwargs.setdefault("network", False)
        self.use_lstm_input = self.network_param["use_lstm_input"]   # (kept name) sequence input
        self.use_dueling = self.network_param["use_dueling"]
        self.use_double = self.network_param["use_double"]
        self.nn_layers = self.network_param["layers"]

        # ---------------- Transformer / neighbor-attention params (safe defaults) ----------------
        self.d_model = int(self.network_param.get("d_model", list(self.nn_layers.values())[0]))
        self.num_heads = int(self.network_param.get("num_heads", 4))
        self.transformer_layers = int(self.network_param.get("transformer_layers", 2))
        self.ff_dim = int(self.network_param.get("ff_dim", max(4 * self.d_model, self.d_model)))
        self.dropout_rate = float(self.network_param.get("dropout_rate", 0.0))

        self.use_neighbor_attention = bool(self.network_param.get("use_neighbor_attention", True))
        self.neighbor_radius = float(self.network_param.get("neighbor_radius", 200.0))
        self.pos_index_start = int(self.network_param.get("pos_index_start", 0))  # x,y start in each user-block

        # ========================== Build Network ============================= #
        if self.use_lstm_input:
            self.inputs_ = tf.placeholder(tf.float32, [None, self.step_size, self.state_size], name='inputs_')
        else:
            self.inputs_ = tf.placeholder(tf.float32, [None, self.state_size], name='inputs_')

        # NEW: user id placeholder (needed for neighbor-attention + consistent training feed)
        self.user_id_ = tf.placeholder(tf.int32, [None], name='user_id')

        self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
        self.one_hot_actions = tf.one_hot(self.actions_, self.action_size)
        self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

        with tf.variable_scope(name):
            with tf.variable_scope("eval_net_scope"):
                self.eval_scope_name = tf.get_variable_scope().name
                self.qvalues = self._create_network()

            with tf.variable_scope("target_net_scope"):
                self.target_scope_name = tf.get_variable_scope().name
                self.target_qvalues = self._create_network()

            # ============== LOSS ============== #
            self.gamma = kwargs.setdefault("gamma", 0.99)
            self.Q = tf.reduce_sum(tf.multiply(self.qvalues, self.one_hot_actions), axis=1)
            self.h_loss = self.Q - self.targetQs_
            if self.hysteretic:
                self.h_loss = tf.where(self.h_loss < 0, self.h_loss / 10.0, self.h_loss)

            self.loss = tf.reduce_mean(tf.square(self.h_loss))
            self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # target network update op
        self.update_target_op = []
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.target_scope_name)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.eval_scope_name)
        for i in range(len(t_params)):
            self.update_target_op.append(tf.assign(t_params[i], e_params[i]))

        # init tensorflow session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        #################### Create Policy ######################
        self.policy_name = kwargs.setdefault("policy", "")
        if self.policy_name == "softmax":
            self.policy = SoftmaxPolicy(nA=self.action_size, temperature=self.temperature, episodes=total_episodes)
        elif self.policy_name == "boltzman":
            self.policy = BoltzmanPolicy(self.action_size, beta=self.beta, explore_start=self.explore_start,
                                         explore_stop=self.explore_stop, decay_rate=self.decay_rate, alpha=self.alpha)
        elif self.policy_name == "eps_greedy":
            self.policy = EpsilonGreedy(eps_init=self.eps, eps_decay=self.eps_decay, nA=self.action_size,
                                        episodes=total_episodes, explore_stop=self.explore_stop)
        else:
            self.policy = GreedyPolicy(nA=self.action_size)

    # -------------------------- Transformer helpers --------------------------

    def _layer_norm(self, x, scope):
        return tf.contrib.layers.layer_norm(x, begin_norm_axis=-1, scope=scope)

    def _split_heads(self, x, num_heads):
        # x: [B, T, D] -> [B, H, T, D/H]
        d = tf.shape(x)[-1]
        depth = tf.cast(d // num_heads, tf.int32)
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], num_heads, depth])
        return tf.transpose(x, [0, 2, 1, 3])

    def _combine_heads(self, x):
        # x: [B, H, T, depth] -> [B, T, H*depth]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])

    def _mha(self, q_in, k_in, v_in, num_heads, scope, mask=None):
        """
        Multi-head attention.
        q_in: [B, Tq, D], k_in/v_in: [B, Tk, D]
        mask: [B, Tk] boolean, True => mask out (set -inf)
        """
        with tf.variable_scope(scope):
            d_model = self.d_model
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads")

            q = tf.layers.dense(q_in, d_model, use_bias=False, name="Wq")
            k = tf.layers.dense(k_in, d_model, use_bias=False, name="Wk")
            v = tf.layers.dense(v_in, d_model, use_bias=False, name="Wv")

            qh = self._split_heads(q, num_heads)  # [B,H,Tq,depth]
            kh = self._split_heads(k, num_heads)  # [B,H,Tk,depth]
            vh = self._split_heads(v, num_heads)  # [B,H,Tk,depth]

            depth = tf.cast(d_model // num_heads, tf.float32)
            logits = tf.matmul(qh, kh, transpose_b=True) / tf.sqrt(depth)  # [B,H,Tq,Tk]

            if mask is not None:
                # mask: True => invalid
                m = tf.cast(mask, tf.float32)  # [B,Tk]
                m = tf.expand_dims(tf.expand_dims(m, 1), 1)  # [B,1,1,Tk]
                logits = logits + (m * -1e9)

            weights = tf.nn.softmax(logits, axis=-1)
            attn = tf.matmul(weights, vh)  # [B,H,Tq,depth]
            attn = self._combine_heads(attn)  # [B,Tq,D]
            out = tf.layers.dense(attn, d_model, use_bias=False, name="Wo")
            return out

    def _ffn(self, x, scope):
        with tf.variable_scope(scope):
            h = tf.layers.dense(x, self.ff_dim, activation=tf.nn.relu, name="fc1")
            h = tf.layers.dense(h, self.d_model, activation=None, name="fc2")
            return h

    def _transformer_encoder(self, x, scope):
        """
        x: [B, T, D]
        """
        with tf.variable_scope(scope):
            for i in range(self.transformer_layers):
                with tf.variable_scope("layer_%d" % i):
                    attn = self._mha(x, x, x, self.num_heads, scope="self_attn", mask=None)
                    x = self._layer_norm(x + attn, scope="ln1")

                    f = self._ffn(x, scope="ffn")
                    x = self._layer_norm(x + f, scope="ln2")
            return x

    def _neighbor_attention(self, last_state_flat, scope):
        """
        last_state_flat: [B, state_size]
        Interprets last_state_flat as concatenation of per-user blocks: [num_users, per_user_dim]
        Uses x,y positions within each user-block to mask neighbors beyond neighbor_radius.
        Returns: [B, d_model]
        """
        with tf.variable_scope(scope):
            state_size = self.state_size
            num_users = self.num_users

            # If state cannot be reshaped cleanly, disable safely (returns zeros).
            if state_size % num_users != 0:
                return tf.zeros([tf.shape(last_state_flat)[0], self.d_model], dtype=tf.float32)

            per_user_dim = state_size // num_users
            s = tf.reshape(last_state_flat, [tf.shape(last_state_flat)[0], num_users, per_user_dim])  # [B,N,Du]

            # If no room for x,y indices, disable safely.
            if per_user_dim < (self.pos_index_start + 2):
                return tf.zeros([tf.shape(last_state_flat)[0], self.d_model], dtype=tf.float32)

            pos = s[:, :, self.pos_index_start:self.pos_index_start + 2]  # [B,N,2]

            # gather ego block by user_id_
            batch_ids = tf.range(tf.shape(s)[0], dtype=tf.int32)
            gather_idx = tf.stack([batch_ids, self.user_id_], axis=1)  # [B,2]
            ego_block = tf.gather_nd(s, gather_idx)   # [B,Du]
            ego_pos = tf.gather_nd(pos, gather_idx)   # [B,2]

            # compute distances
            diff = pos - tf.expand_dims(ego_pos, axis=1)  # [B,N,2]
            dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))  # [B,N]

            # mask neighbors farther than radius, but never mask ego itself
            mask = dist > self.neighbor_radius  # [B,N]
            ego_onehot = tf.one_hot(self.user_id_, depth=num_users, dtype=tf.bool)  # [B,N]
            mask = tf.logical_and(mask, tf.logical_not(ego_onehot))

            # embed user blocks
            kv = tf.layers.dense(s, self.d_model, activation=None, name="kv_embed")     # [B,N,D]
            q = tf.layers.dense(tf.expand_dims(ego_block, 1), self.d_model, activation=None, name="q_embed")  # [B,1,D]

            # attention (q attends to all users within radius)
            ctx = self._mha(q, kv, kv, self.num_heads, scope="nei_attn", mask=mask)  # [B,1,D]
            ctx = tf.squeeze(ctx, axis=1)  # [B,D]
            return ctx

    # -------------------------- Network --------------------------

    def _create_network(self):
        """
        Creates a transformer-based network for sequential inputs.
        Returns Q-values [B, action_size]
        """
        hidden_size = list(self.nn_layers.values())

        if self.use_lstm_input:
            # inputs_: [B, step_size, state_size]
            x = self.inputs_

            # project to d_model
            x = tf.layers.dense(x, self.d_model, activation=None, name="in_proj")

            # trainable positional embedding (fixed step_size)
            pos = tf.get_variable("pos_emb", shape=[self.step_size, self.d_model],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
            x = x + tf.expand_dims(pos, axis=0)  # [B,T,D]

            # transformer encoder over time
            x = self._transformer_encoder(x, scope="time_transformer")  # [B,T,D]

            temporal_summary = x[:, -1, :]  # [B,D]

            if self.use_neighbor_attention:
                last_flat = self.inputs_[:, -1, :]  # [B,state_size]
                neighbor_ctx = self._neighbor_attention(last_flat, scope="neighbor_block")  # [B,D]
            else:
                neighbor_ctx = tf.zeros_like(temporal_summary)

            # fuse temporal + neighbor context
            fused = tf.concat([temporal_summary, neighbor_ctx], axis=1)  # [B,2D]
            fused = self._layer_norm(fused, scope="fuse_ln")

            # head (kept simple, consistent with original dense style)
            h1 = tf.layers.dense(fused, hidden_size[0], activation=tf.nn.relu, name="head_fc1")
            h1 = self._layer_norm(h1, scope="head_ln1")

            if len(hidden_size) > 1:
                h2 = tf.layers.dense(h1, hidden_size[1], activation=tf.nn.relu, name="head_fc2")
                h2 = self._layer_norm(h2, scope="head_ln2")
            else:
                h2 = h1

            if len(hidden_size) > 2:
                h3 = tf.layers.dense(h2, hidden_size[2], activation=tf.nn.relu, name="head_fc3")
                h3 = self._layer_norm(h3, scope="head_ln3")
                output = tf.layers.dense(h3, self.action_size, activation=None, name="q_out")
            else:
                output = tf.layers.dense(h2, self.action_size, activation=None, name="q_out")

            return output

        else:
            # Non-sequence mode (kept compatible)
            h1 = tf.layers.dense(self.inputs_, hidden_size[0], activation=tf.nn.relu, name="fc1")
            h1 = self._layer_norm(h1, scope="ln_fc1")
            if len(hidden_size) > 1:
                h2 = tf.layers.dense(h1, hidden_size[1], activation=tf.nn.relu, name="fc2")
                h2 = self._layer_norm(h2, scope="ln_fc2")
            else:
                h2 = h1
            if len(hidden_size) > 2:
                h3 = tf.layers.dense(h2, hidden_size[2], activation=tf.nn.relu, name="fc3")
                h3 = self._layer_norm(h3, scope="ln_fc3")
                output = tf.layers.dense(h3, self.action_size, activation=None, name="q_out")
            else:
                output = tf.layers.dense(h2, self.action_size, activation=None, name="q_out")
            return output

    # -------------------------- Inference --------------------------

    def infer_action(self, user, state_vector, episode, policy="boltzman"):
        """
        Same signature as original infer_action (main code calls this per user).
        Original feed behavior is preserved, but now we also feed user_id_.
        """
        if policy == "explore":
            return np.random.randint(self.action_size)

        if self.use_lstm_input:
            feed = {
                self.inputs_: state_vector[:, user].reshape(1, self.step_size, self.state_size),
                self.user_id_: np.array([user], dtype=np.int32)
            }
        else:
            feed = {
                self.inputs_: state_vector[-1:, user].reshape(1, self.state_size),
                self.user_id_: np.array([user], dtype=np.int32)
            }

        Qs = self.sess.run(self.qvalues, feed_dict=feed)

        if policy == "greedy":
            action = np.argmax(Qs, axis=1)
        else:
            action = self.policy.action(Qs, episode)

        return action

    def set_eps(self, eps):
        self.policy.set_epsilon(eps)

    def get_eps(self):
        return self.policy.get_epsilon()

    # -------------------------- Training --------------------------

    def train(self, sample_buffer, time_step):
        """
        Same training loop as original, but now feeds user_id_ for each flattened sample row.
        """
        n_batches = self.n_batch
        for k in range(n_batches):

            if self.use_lstm_input:
                batch = sample_buffer.sample(self.batch_size, self.step_size)
            else:
                batch = sample_buffer.sample(self.batch_size, 1)

            states = self.get_states_user(batch)        # [U,B,T,S]
            actions = self.get_actions_user(batch)      # [U,B,T] or [U,B,1]
            rewards = self.get_rewards_user(batch)      # [U,B,T] or [U,B,1]
            next_states = self.get_next_states_user(batch)

            num_users = states.shape[0]
            bsz = states.shape[1]

            if self.use_lstm_input:
                states = np.reshape(states, [-1, states.shape[2], states.shape[3]])         # [U*B,T,S]
                actions = np.reshape(actions, [-1, actions.shape[2]])                       # [U*B,T]
                rewards = np.reshape(rewards, [-1, rewards.shape[2]])                       # [U*B,T]
                next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])
            else:
                states = np.reshape(states, [-1, states.shape[3]])                          # [U*B,S]
                actions = np.reshape(actions, [-1])                                         # [U*B]
                rewards = np.reshape(rewards, [-1])                                         # [U*B]
                next_states = np.reshape(next_states, [-1, next_states.shape[3]])

            # Build user_ids aligned with reshape order (user-major blocks)
            user_ids = np.repeat(np.arange(num_users, dtype=np.int32), bsz)

            targets = self._calc_target(rewards, next_states, user_ids=user_ids)

            if self.use_lstm_input:
                acts = actions[:, -1]
            else:
                acts = actions

            loss, _ = self.sess.run(
                [self.loss, self.opt],
                feed_dict={
                    self.inputs_: states,
                    self.user_id_: user_ids,
                    self.targetQs_: targets,
                    self.actions_: acts
                }
            )

        if (time_step + 1) % self.target_update == 0:
            self.sess.run(self.update_target_op)

    def _calc_target(self, rewards, next_states, user_ids=None):
        """
        Calculates TD target. Now includes user_id_ feed for transformer+neighbor attention.
        """
        n = len(rewards)

        feed = {self.inputs_: next_states}
        if user_ids is None:
            user_ids = np.zeros((n,), dtype=np.int32)
        feed[self.user_id_] = user_ids

        if self.use_double:
            t_qvalues, qvalues = self.sess.run([self.target_qvalues, self.qvalues], feed_dict=feed)
            act = np.argmax(qvalues, axis=1)
            next_value = t_qvalues[np.arange(n), act]
        else:
            t_qvalues = self.sess.run(self.target_qvalues, feed_dict=feed)
            next_value = np.max(t_qvalues, axis=1)

        if self.use_lstm_input:
            target = rewards[:, -1] + self.gamma * next_value
        else:
            target = rewards + self.gamma * next_value

        return target

    # -------------------------- Batch utilities (unchanged) --------------------------

    def get_states_user(self, batch):
        states = []
        for user in range(self.num_users):
            states_per_user = []
            for each in batch:
                states_per_batch = []
                for step_i in each:
                    try:
                        states_per_step = step_i[0][user]
                    except IndexError:
                        print(step_i)
                        print("-----------")
                        print("error")
                        sys.exit()
                    states_per_batch.append(states_per_step)
                states_per_user.append(states_per_batch)
            states.append(states_per_user)
        return np.array(states)

    def get_actions_user(self, batch):
        actions = []
        for user in range(self.num_users):
            actions_per_user = []
            for each in batch:
                actions_per_batch = []
                for step_i in each:
                    actions_per_step = step_i[1][user]
                    actions_per_batch.append(actions_per_step)
                actions_per_user.append(actions_per_batch)
            actions.append(actions_per_user)
        return np.array(actions)

    def get_rewards_user(self, batch):
        rewards = []
        for user in range(self.num_users):
            rewards_per_user = []
            for each in batch:
                rewards_per_batch = []
                for step_i in each:
                    rewards_per_step = step_i[2][user]
                    rewards_per_batch.append(rewards_per_step)
                rewards_per_user.append(rewards_per_batch)
            rewards.append(rewards_per_user)
        return np.array(rewards)

    def get_next_states_user(self, batch):
        next_states = []
        for user in range(self.num_users):
            next_states_per_user = []
            for each in batch:
                next_states_per_batch = []
                for step_i in each:
                    next_states_per_step = step_i[3][user]
                    next_states_per_batch.append(next_states_per_step)
                next_states_per_user.append(next_states_per_batch)
            next_states.append(next_states_per_user)
        return np.array(next_states)

    def sample(self):
        return np.random.randint(self.action_size)

    def save_model(self, dir_name, slot, simulation):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(dir_name, self.name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
        saver = tf.train.Saver(model_vars)
        saver.save(self.sess, os.path.join(dir_name, ("sim_%d_%d") % (simulation, slot)))

    def load_model(self, dir_name, epoch=0, name=None):
        if name is None or name == self.name:
            dir_name = os.path.join(dir_name, self.name)
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            saver = tf.train.Saver(model_vars)
            saver.restore(self.sess, os.path.join(dir_name, ("sim_0_%d") % epoch))
        else:
            backup_graph = tf.get_default_graph()
            kv_dict = {}

            with tf.Graph().as_default(), tf.Session() as sess:
                tf.train.import_meta_graph(os.path.join(dir_name, name, (self.subclass_name + "_%d") % epoch + ".meta"))
                dir_name = os.path.join(dir_name, name)
                model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(model_vars)
                saver.restore(sess, os.path.join(dir_name, (self.subclass_name + "_%d") % epoch))
                for item in tf.global_variables():
                    kv_dict[item.name] = sess.run(item)

            backup_graph.as_default()
            model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)
            for item in model_vars:
                old_name = item.name.replace(self.name, name)
                self.sess.run(tf.assign(item, kv_dict[old_name]))
try:
    _tf_v1_ok = hasattr(tf, "reset_default_graph") and hasattr(tf, "Session") and hasattr(tf, "contrib")
except Exception:
    _tf_v1_ok = False
if not _tf_v1_ok:
    class DRQN:
        def __init__(self, env, name="DeepRQN", total_episodes=4000, **kwargs):
            self.env = env
            self.name = name
            self.num_users = env.get_total_users()
            self.action_size = env.get_action_space()
            self.state_size = env.get_state_space()
            self.learning_rate = kwargs.setdefault("learning_rate", 1e-4)
            self.target_update = kwargs.setdefault("target_update", 10)
            self.batch_size = kwargs.setdefault("batch_size", 64)
            self.step_size = kwargs.setdefault("step_size", 5)
            self.n_batch = kwargs.setdefault("n_batch", 1)
            self.gamma = kwargs.setdefault("gamma", 0.99)
            self.policy_name = kwargs.setdefault("policy", "")
            self.eps = kwargs.setdefault("eps_init", 1.0)
            self.eps_decay = kwargs.setdefault("eps_decay", 0.995)
            self.temperature = kwargs.setdefault("temperature", 0.01)
            self.network_param = kwargs.setdefault("network", {})
            self.use_lstm_input = bool(self.network_param.get("use_lstm_input", True))
            layers = self.network_param.get("layers", {"l1": 64, "l2": 32})
            self.d_model = int(self.network_param.get("d_model", list(layers.values())[0]))
            self.num_heads = int(self.network_param.get("num_heads", 4))
            self.transformer_layers = int(self.network_param.get("transformer_layers", 2))
            self.ff_dim = int(self.network_param.get("ff_dim", max(4 * self.d_model, self.d_model)))
            self.dropout_rate = float(self.network_param.get("dropout_rate", 0.0))
            self.use_neighbor_attention = bool(self.network_param.get("use_neighbor_attention", True))
            self.neighbor_radius = float(self.network_param.get("neighbor_radius", 200.0))
            if self.policy_name == "softmax":
                self.policy = SoftmaxPolicy(nA=self.action_size, temperature=self.temperature, episodes=total_episodes)
            elif self.policy_name == "boltzman":
                self.policy = BoltzmanPolicy(self.action_size, beta=kwargs.setdefault("beta", 1),
                                             explore_start=kwargs.setdefault("explore_start", 4),
                                             explore_stop=kwargs.setdefault("explore_stop", 4),
                                             decay_rate=kwargs.setdefault("decay_rate", 4),
                                             alpha=kwargs.setdefault("alpha", 1))
            elif self.policy_name == "eps_greedy":
                self.policy = EpsilonGreedy(eps_init=self.eps, eps_decay=self.eps_decay, nA=self.action_size,
                                            episodes=total_episodes, explore_stop=kwargs.setdefault("explore_stop", 4))
            else:
                self.policy = GreedyPolicy(nA=self.action_size)
            if self.use_lstm_input:
                inp = tf.keras.Input(shape=(self.step_size, self.state_size))
                x = tf.keras.layers.Dense(self.d_model, activation=None)(inp)
                pos = tf.keras.layers.Embedding(input_dim=self.step_size, output_dim=self.d_model)(
                    tf.range(self.step_size))
                pos = tf.expand_dims(pos, axis=0)
                x = x + pos
                for _ in range(self.transformer_layers):
                    attn = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads)(x, x)
                    x = tf.keras.layers.LayerNormalization()(x + attn)
                    f = tf.keras.layers.Dense(self.ff_dim, activation="relu")(x)
                    f = tf.keras.layers.Dense(self.d_model)(f)
                    x = tf.keras.layers.LayerNormalization()(x + f)
                x = x[:, -1, :]
            else:
                inp = tf.keras.Input(shape=(self.state_size,))
                x = tf.keras.layers.Dense(self.d_model, activation="relu")(inp)
                x = tf.keras.layers.LayerNormalization()(x)
            h1 = tf.keras.layers.Dense(layers.get("l1", 64), activation="relu")(x)
            h1 = tf.keras.layers.LayerNormalization()(h1)
            if len(layers) > 1:
                h2 = tf.keras.layers.Dense(list(layers.values())[1], activation="relu")(h1)
                h2 = tf.keras.layers.LayerNormalization()(h2)
            else:
                h2 = h1
            out = tf.keras.layers.Dense(self.action_size, activation=None)(h2)
            self.model = tf.keras.Model(inputs=inp, outputs=out)
            self.target_model = tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        def infer_action(self, user, state_vector, episode, policy="boltzman"):
            if policy == "explore":
                return int(np.random.randint(self.action_size))
            if self.use_lstm_input:
                inp = state_vector[:, user].reshape(1, self.step_size, self.state_size)
            else:
                inp = state_vector[-1:, user].reshape(1, self.state_size)
            Qs = self.model.predict(inp, verbose=0)
            if policy == "greedy":
                action = np.argmax(Qs, axis=1)
            else:
                action = self.policy.action(Qs, episode)
            return action
        def set_eps(self, eps):
            try:
                self.policy.set_epsilon(eps)
            except Exception:
                pass
        def get_eps(self):
            try:
                return self.policy.get_epsilon()
            except Exception:
                return self.eps
        def train(self, sample_buffer, time_step):
            n_batches = self.n_batch
            for _ in range(n_batches):
                if self.use_lstm_input:
                    batch = sample_buffer.sample(self.batch_size, self.step_size)
                else:
                    batch = sample_buffer.sample(self.batch_size, 1)
                states = self.get_states_user(batch)
                actions = self.get_actions_user(batch)
                rewards = self.get_rewards_user(batch)
                next_states = self.get_next_states_user(batch)
                if self.use_lstm_input:
                    states = np.reshape(states, [-1, states.shape[2], states.shape[3]])
                    actions = np.reshape(actions, [-1, actions.shape[2]])
                    rewards = np.reshape(rewards, [-1, rewards.shape[2]])
                    next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])
                else:
                    states = np.reshape(states, [-1, states.shape[3]])
                    actions = np.reshape(actions, [-1])
                    rewards = np.reshape(rewards, [-1])
                    next_states = np.reshape(next_states, [-1, next_states.shape[3]])
                q_next = self.target_model.predict(next_states, verbose=0)
                next_value = np.max(q_next, axis=1)
                if self.use_lstm_input:
                    targets = rewards[:, -1] + self.gamma * next_value
                    acts = actions[:, -1]
                else:
                    targets = rewards + self.gamma * next_value
                    acts = actions
                with tf.GradientTape() as tape:
                    q_pred = self.model(states, training=True)
                    idx = tf.stack([tf.range(tf.shape(q_pred)[0]), tf.cast(acts, tf.int32)], axis=1)
                    q_taken = tf.gather_nd(q_pred, idx)
                    diff = q_taken - tf.cast(targets, tf.float32)
                    loss = tf.reduce_mean(tf.square(diff))
                    rmse = tf.sqrt(tf.reduce_mean(tf.square(diff)))
                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            if (time_step + 1) % self.target_update == 0:
                self.target_model.set_weights(self.model.get_weights())
        def sample(self):
            return int(np.random.randint(self.action_size))
        def save_model(self, dir_name, slot, simulation):
            try:
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                target_dir = os.path.join(dir_name, self.name)
                if not os.path.exists(target_dir):
                    os.mkdir(target_dir)
                path = os.path.join(target_dir, "sim_%d_%d.weights.h5" % (simulation, slot))
                self.model.save_weights(path)
            except Exception:
                pass
        def load_model(self, dir_name, epoch=0, name=None):
            try:
                if name is None:
                    target_dir = os.path.join(dir_name, self.name)
                else:
                    target_dir = os.path.join(dir_name, name)
                path = os.path.join(target_dir, "sim_0_%d.weights.h5" % epoch)
                self.model.load_weights(path)
                self.target_model.set_weights(self.model.get_weights())
            except Exception:
                pass
        def get_states_user(self, batch):
            states = []
            for user in range(self.num_users):
                states_per_user = []
                for each in batch:
                    states_per_batch = []
                    for step_i in each:
                        states_per_step = step_i[0][user]
                        states_per_batch.append(states_per_step)
                    states_per_user.append(states_per_batch)
                states.append(states_per_user)
            return np.array(states)
        def get_actions_user(self, batch):
            actions = []
            for user in range(self.num_users):
                actions_per_user = []
                for each in batch:
                    actions_per_batch = []
                    for step_i in each:
                        actions_per_step = step_i[1][user]
                        actions_per_batch.append(actions_per_step)
                    actions_per_user.append(actions_per_batch)
                actions.append(actions_per_user)
            return np.array(actions)
        def get_rewards_user(self, batch):
            rewards = []
            for user in range(self.num_users):
                rewards_per_user = []
                for each in batch:
                    rewards_per_batch = []
                    for step_i in each:
                        rewards_per_step = step_i[2][user]
                        rewards_per_batch.append(rewards_per_step)
                    rewards_per_user.append(rewards_per_batch)
                rewards.append(rewards_per_user)
            return np.array(rewards)
        def get_next_states_user(self, batch):
            next_states = []
            for user in range(self.num_users):
                next_states_per_user = []
                for each in batch:
                    next_states_per_batch = []
                    for step_i in each:
                        next_states_per_step = step_i[3][user]
                        next_states_per_batch.append(next_states_per_step)
                    next_states_per_user.append(next_states_per_batch)
                next_states.append(next_states_per_user)
            return np.array(next_states)
