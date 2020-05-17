from Blackjack import BlackjackEnv
import numpy as np
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.models.preprocessors import Preprocessor
from tensorflow.keras.layers import Add, Concatenate
from ray.rllib.utils.annotations import override
from ray.tune.registry import register_env
from ray.rllib.models.modelv2 import ModelV2

#lstm self defined model
class MyKerasModel(RecurrentTFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 hiddens_size=512,
                 cell_size=256):
        super(MyKerasModel, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        self.cell_size = cell_size


        ## inputs
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]))
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ))
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size, ))
        seq_in = tf.keras.layers.Input(shape=(), dtype=tf.int32)

        ##actor network
        # Send to LSTM cell
        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True,
            name="lstm")(
            inputs=input_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])
        action_h = tf.keras.layers.Dense(hiddens_size)(lstm_out)
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(action_h)

        ##value network
        v_lstm_out, v_state_h, v_state_c = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True,
            name="v_lstm")(
            inputs=input_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c])
        value_h = tf.keras.layers.Dense(hiddens_size)(v_lstm_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(value_h)


        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[logits, values, state_h, state_c])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h, c = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h, c]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("my_model", MyKerasModel)


analysis=tune.run(PPOTrainer, stop={"timesteps_total": 100000},
                  config={
                      "env": BlackjackEnv,
                      "gamma": 0.99,
                      "num_workers": 1,
                      "num_envs_per_worker": 8,
                      "entropy_coeff": 0.001,
                      "num_sgd_iter": 5,
                      "vf_loss_coeff": 1e-5,
                      "lr":tune.grid_search([0.0001,0.0005,0.00001,0.00005]),
                      "model":
                          {
                             "custom_model": "my_model",
                             "max_seq_len": 3,
                          }


                  })
print(analysis.get_best_config("episode_reward_mean"))