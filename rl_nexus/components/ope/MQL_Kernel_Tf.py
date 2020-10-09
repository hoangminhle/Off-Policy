import tensorflow as tf
import numpy as np
import time
from rl_nexus.components.ope.common.base_tf_estimator import Basic_Alg
from rl_nexus.utils.ope_utils import Q_Model_Tf
# from Model.Basic_Alg_Class import Basic_Alg

class MQL(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, seed, 
                       norm, model_weights, target_temp, med_dist,
                       hidden_layers_p, activation_p, hidden_layers=[32, 32], scope='mql',
                       lr=5e-3, reg_factor=0, gamma=0.999):
        super().__init__(scope)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.reg_factor = reg_factor
        self.hidden_layers = hidden_layers
        self.seed = seed

        self.med_dist = med_dist
        # self.q_net = q_net
        self.norm = norm

        #XXX debug
        self.debug_q = {}

        self.trainable_vars = []
        self._session = tf.compat.v1.Session()
        self._session.__enter__()
        with self._session.as_default():
            self.q_net = Q_Model_Tf(obs_dim, act_dim, hidden_layers = hidden_layers_p, temperature=target_temp, seed = self.seed,\
                activation = activation_p)

        self.build_graph()
        self.build_estimation_graph()
        self.create_loss_func()
        
        self._session.run([tf.compat.v1.variables_initializer(self.trainable_vars)])
        self.q_net.load_weight(model_weights)
        # self._session.run(tf.global_variables_initializer())
        # tf.get_default_session().run(
        #     [tf.variables_initializer(self.trainable_vars)]
        # )
    
    def build_graph(self):        
        ''' firs sample '''       
        self.rew_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
        self.obs_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1])
        self.obs_act = tf.concat([self.obs_ph, tf.cast(self.act_ph, tf.float32)], axis=1)
        self.q = self.create_value_func(self.obs_ph, self.act_ph, func_type='q', reuse=False)

        self.next_obs_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])  
        self.v_next = self.create_value_func(self.next_obs_ph, None, func_type='v', reuse=True)

        ''' second sample '''
        self.rew_ph_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
        self.obs_ph_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.act_ph_2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1])
        self.obs_act_2 = tf.concat([self.obs_ph_2, tf.cast(self.act_ph_2, tf.float32)], axis=1)
        self.q_2 = self.create_value_func(self.obs_ph_2, self.act_ph_2, func_type='q', reuse=True)

        self.next_obs_ph_2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.v_next_2 = self.create_value_func(self.next_obs_ph_2, None, func_type='v', reuse=True)

    def create_value_func(self, obs_tf, act_tf, *, func_type, reuse=False, normalize=True):
        if func_type == 'v':
            if self.norm['type'] is not None:
                org_obs = obs_tf * self.norm['scale'] + self.norm['shift']
            else:
                org_obs = obs_tf
            prob_mask = self.q_net.build_prob(org_obs, split=True)
            
            with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
                x = tf.concat([obs_tf, tf.zeros([tf.shape(input=obs_tf)[0], 1])], axis=1)
                for h in self.hidden_layers:
                    x = tf.compat.v1.layers.dense(x, h, activation=tf.nn.relu)
                q0 = tf.compat.v1.layers.dense(x, 1, activation=None, 
                                kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1.)),
                                bias_regularizer=tf.keras.regularizers.l2(0.5 * (1.)))
            
            with tf.compat.v1.variable_scope(self.scope, reuse=True):
                x = tf.concat([obs_tf, tf.ones([tf.shape(input=obs_tf)[0], 1])], axis=1)
                for h in self.hidden_layers:
                    x = tf.compat.v1.layers.dense(x, h, activation=tf.nn.relu)
                q1 = tf.compat.v1.layers.dense(x, 1, activation=None, 
                                kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1.)),
                                bias_regularizer=tf.keras.regularizers.l2(0.5 * (1.)))
            value = q0 * prob_mask[0] + q1 * prob_mask[1]
        else:
            
            with tf.compat.v1.variable_scope(self.scope, reuse=reuse):
                x = tf.concat([obs_tf, tf.cast(act_tf, tf.float32)], axis=1)
                for h in self.hidden_layers:
                    x = tf.compat.v1.layers.dense(x, h, activation=tf.nn.relu)
                q = tf.compat.v1.layers.dense(x, 1, activation=None, 
                                kernel_regularizer=tf.keras.regularizers.l2(0.5 * (1.)),
                                bias_regularizer=tf.keras.regularizers.l2(0.5 * (1.)))

                if reuse == False:
                    self.trainable_vars += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)                    
                value = q            
        return value

    def build_estimation_graph(self):
        self.init_obs_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.value_estimation = tf.reduce_mean(input_tensor=self.create_value_func(self.init_obs_ph, None, func_type='v', reuse=True))

    def create_loss_func(self):
        error = self.rew_ph + self.gamma * self.v_next - self.q
        error_2 = self.rew_ph_2 + self.gamma * self.v_next_2 - self.q_2

        diff = tf.expand_dims(self.obs_act, 1) - tf.expand_dims(self.obs_act_2, 0)
        K = tf.exp(-tf.reduce_sum(input_tensor=tf.square(diff), axis=-1) / 2.0 / self.med_dist ** 2)

        all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        self.loss = tf.matmul(tf.matmul(tf.transpose(a=error), K), error_2)

        self.reg_loss = self.reg_factor * tf.reduce_sum(input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, self.scope))
        self.debug_q.update({'reg loss': self.reg_loss})
        self.loss += self.reg_loss
        self.loss = tf.squeeze(self.loss)

        sample_num = tf.cast(tf.shape(input=K)[0] * tf.shape(input=K)[1], tf.float32)
        self.loss /= sample_num

        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.train_op = self.opt.minimize(self.loss, var_list=all_vars)

        self.trainable_vars += self.opt.variables()

    def train(self, data):
        # debug, loss, _ = tf.get_default_session().run(
        debug, loss, _ = self._session.run(            
            [self.debug_q, self.loss, self.train_op],
            feed_dict={
                self.obs_ph: data['obs_1'],    
                self.obs_ph_2: data['obs_2'],
                self.next_obs_ph: data['next_obs_1'],
                self.next_obs_ph_2: data['next_obs_2'],
                self.act_ph: data['act_1'],
                self.act_ph_2: data['act_2'],                
                self.rew_ph: data['rew_1'],
                self.rew_ph_2: data['rew_2'],
                self.q_net.tau_ph: self.q_net.temperature,
            }
        )

        return debug, loss

    def evaluation(self, init_obs):
        # value = tf.get_default_session().run(
        value = self._session.run(
            self.value_estimation,
            feed_dict={
                self.init_obs_ph: init_obs,
                self.q_net.tau_ph: self.q_net.temperature,
            }
        )
        return value
    def close(self):
        self._session.close()
