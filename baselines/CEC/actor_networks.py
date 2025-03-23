from typing import Sequence, NamedTuple, Any
import flax.linen as nn
from graph_layer import GATLayer, GCNLayer, make_graph
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
import numpy as np
import distrax
from pdb import set_trace as T
import functools
from typing import Sequence, NamedTuple, Any, Dict



class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        lstm_state = carry
        ins, resets = x
        
        # Reset LSTM state on episode boundaries
        lstm_state = jax.tree_map(
            lambda x: jnp.where(resets[:, np.newaxis], jnp.zeros_like(x), x),
            lstm_state
        )
        
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[-1])(lstm_state, ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.OptimizedLSTMCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["GRAPH_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
            # if self.config["ENV_NAME"] == "overcooked":
            #     reshaped_obs = obs.reshape(-1, 7,7,26)
            # else:
            #     reshaped_obs = obs.reshape(-1, 5,5,3)
            reshaped_obs = obs.reshape(-1, *self.config["obs_dim"])
            # # use 2 conv nets
            # embedding = nn.Conv(
            #     features=self.config["FC_DIM_SIZE"]*2,
            #     kernel_size=(2, 2),
            #     kernel_init=orthogonal(np.sqrt(2)),
            #     bias_init=constant(0.0),
            # )(reshaped_obs)
            # embedding = nn.relu(embedding)
            # embedding = nn.Conv(
            #     features=self.config["FC_DIM_SIZE"],
            #     kernel_size=(2, 2),
            #     kernel_init=orthogonal(np.sqrt(2)),
            #     bias_init=constant(0.0),
            # )(embedding)
            # embedding = nn.relu(embedding)

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

            # reshaped_obs = obs.reshape(-1, flattened_obs_dim)
            # reshaped_agent_positions = agent_positions.reshape((-1, 2, 2))
            # make_graph_fn = make_graph_overcooked if self.config["ENV_NAME"] == "overcooked" else make_graph_toy_coop
            # node_feats, adj_mat = jax.vmap(make_graph_fn)(reshaped_obs, reshaped_agent_positions)
            # embedding = GATLayer(self.config["FC_DIM_SIZE"], num_heads=2)(node_feats, adj_mat)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(embedding)
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        #########
        # Actor
        #########
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1
            # actor_mean = nn.Dense(
            #     self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
            # )(actor_mean)
            # actor_mean = nn.relu(actor_mean)  # extra layer 2
            # actor_mean = nn.Dense(
            #     self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
            # )(actor_mean)
            # actor_mean = nn.relu(actor_mean)  # extra layer 3
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class ActorCriticE3T(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, agent_positions = x
        if self.config["GRAPH_NET"]:
            batch_size, num_envs, flattened_obs_dim = obs.shape
            # if self.config["ENV_NAME"] == "overcooked":
            #     reshaped_obs = obs.reshape(-1, 7,7,26)
            # else:
            #     reshaped_obs = obs.reshape(-1, 5,5,3)
            reshaped_obs = obs.reshape(-1, *self.config["obs_dim"])
            # # use 2 conv nets
            # embedding = nn.Conv(
            #     features=self.config["FC_DIM_SIZE"]*2,
            #     kernel_size=(2, 2),
            #     kernel_init=orthogonal(np.sqrt(2)),
            #     bias_init=constant(0.0),
            # )(reshaped_obs)
            # embedding = nn.relu(embedding)
            # embedding = nn.Conv(
            #     features=self.config["FC_DIM_SIZE"],
            #     kernel_size=(2, 2),
            #     kernel_init=orthogonal(np.sqrt(2)),
            #     bias_init=constant(0.0),
            # )(embedding)
            # embedding = nn.relu(embedding)

            embedding = nn.Conv(
                features=64 if "9" in self.config['layout_name'] else 2 * self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(reshaped_obs)
            embedding = nn.relu(embedding)
            embedding = nn.Conv(
                features=32 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"],
                kernel_size=(2, 2),
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

            # reshaped_obs = obs.reshape(-1, flattened_obs_dim)
            # reshaped_agent_positions = agent_positions.reshape((-1, 2, 2))
            # make_graph_fn = make_graph_overcooked if self.config["ENV_NAME"] == "overcooked" else make_graph_toy_coop
            # node_feats, adj_mat = jax.vmap(make_graph_fn)(reshaped_obs, reshaped_agent_positions)
            # embedding = GATLayer(self.config["FC_DIM_SIZE"], num_heads=2)(node_feats, adj_mat)

            embedding = embedding.reshape((batch_size, num_envs, -1))
        else:
            embedding = obs

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)
        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(embedding)
        # embedding = nn.relu(embedding)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"] * 2 if "9" in self.config['layout_name'] else self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        #########
        # Model of other agent
        #########
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.leaky_relu(prediction_other)
        prediction_other = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = nn.tanh(prediction_other)
        prediction_other = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(prediction_other)
        prediction_other = prediction_other / jnp.sqrt(jnp.sum(prediction_other**2, axis=-1, keepdims=True) + 1e-10)  # L2 normalization
        other_pi = distrax.Categorical(logits=prediction_other)

        #########
        # Actor
        #########
        actor_embedding = jnp.concatenate([embedding, prediction_other], axis=-1)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] , kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            actor_mean
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        if self.config["ENV_NAME"] == "overcooked":
            actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"] // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                actor_mean
            )
            actor_mean = nn.relu(actor_mean)  # extra layer 1
            # actor_mean = nn.Dense(
            #     self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
            # )(actor_mean)
            # actor_mean = nn.relu(actor_mean)  # extra layer 2
            # actor_mean = nn.Dense(
            #     self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
            # )(actor_mean)
            # actor_mean = nn.relu(actor_mean)  # extra layer 3
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)        

        pi = distrax.Categorical(logits=actor_mean)

        #########
        # Critic
        #########
        critic = nn.Dense(self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic
        )
        critic = nn.relu(critic)
        if self.config["ENV_NAME"] == "overcooked":
            critic = nn.Dense(self.config["FC_DIM_SIZE"] * 3 // 4, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 1
            critic = nn.Dense(self.config["FC_DIM_SIZE"] // 2, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                critic
            )
            critic = nn.relu(critic)  # extra layer 2
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), other_pi


class GraphActor(nn.Module):
    action_dim: Sequence[int]
    observation_shape: tuple
    activation: str = "tanh"
    model_other_agent: bool = False
    e3t_baseline: bool = False
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, actor_hidden_state=None, critic_hidden_state=None, previous_state_action=None, observation_shape=None):
        '''
        Takes flattened observation as input (h * w * c)
        '''
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if observation_shape is None:
            observation_shape = self.observation_shape
        
        
        if len(x.shape) == 1:  # need to add batch dim if it's not already there
            x = x[None]
        
        # process obs
        partial_make_graph = lambda x: make_graph(x, observation_shape)
        make_whole_graph = jax.vmap(partial_make_graph)
        graph_vals, graph_edges = make_whole_graph(x)

        # feature detection 1
        actor_mean = GATLayer(c_out=32, num_heads=4, concat_heads=True)(graph_vals, graph_edges)
        actor_mean = actor_mean.reshape((x.shape[0], -1)) 

        if self.model_other_agent:
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(actor_mean)  # batch size by 128
            other_actor_mean = activation(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = activation(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_pi = distrax.Categorical(logits=other_actor_mean)

        elif self.e3t_baseline:  # condition on other agent action prediction and our own obs, train separate network
            b_size, num_time = previous_state_action['action'].shape[0], previous_state_action['action'].shape[1]
            past_graphs = jax.vmap(make_whole_graph)(previous_state_action['obs'].reshape((b_size, num_time, -1)))
            
            batch_graph = nn.vmap(
                GATLayer,
                in_axes=1, out_axes=1,
                variable_axes={'params': 1},
                split_rngs={'params': True}
            )

            other_graph_vals = batch_graph(c_out=16, num_heads=2, concat_heads=True)(*past_graphs)
            other_graph_vals = other_graph_vals.reshape((b_size, num_time, -1)) 
            
            embeddings = nn.Embed(num_embeddings=self.action_dim, features=128)(previous_state_action['action'].astype(jnp.int32))
            embeddings = embeddings.reshape((b_size, num_time, -1))  # remove extraneous dimension
            other_actor_mean = jnp.concatenate([other_graph_vals, embeddings], axis=-1)  # concatenate along feature dimension

            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            
            other_actor_mean = other_actor_mean.reshape((b_size, -1))  # concatenate 5 timepoints
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.tanh(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)

            other_pi = distrax.Categorical(logits=other_actor_mean)
            actor_mean = jnp.concatenate([actor_mean, other_actor_mean], axis=-1)

        # action layer 1
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)  # batch size by 128
        actor_mean = activation(actor_mean)
        # action layer 2
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # action output
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        if not (self.model_other_agent or self.e3t_baseline):
            other_pi = distrax.Categorical(logits=jnp.zeros_like(actor_mean))  # dummy value

        # feature detection 1
        critic = GATLayer(c_out=32, num_heads=4, concat_heads=True)(graph_vals, graph_edges)
        critic = critic.reshape((x.shape[0], -1))
        # critic layer 1
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic layer 2
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic output
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_hidden_state, critic_hidden_state, other_pi
    
class GraphLstmActor(nn.Module):
    action_dim: Sequence[int]
    observation_shape: tuple
    activation: str = "tanh"
    model_other_agent: bool = False
    e3t_baseline: bool = False
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, actor_hidden_state=None, critic_hidden_state=None, previous_state_action=None, observation_shape=None): 
        '''
        Takes flattened observation as input (h * w * c)
        '''
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if observation_shape is None:
            observation_shape = self.observation_shape
        
        
        if len(x.shape) == 1:  # need to add batch dim if it's not already there
            x = x[None]
        
        # process obs
        partial_make_graph = lambda x: make_graph(x, observation_shape)
        make_whole_graph = jax.vmap(partial_make_graph)
        graph_vals, graph_edges = make_whole_graph(x)

        # feature detection 1
        actor_mean = GATLayer(c_out=32, num_heads=4, concat_heads=True)(graph_vals, graph_edges)
        actor_mean = actor_mean.reshape((x.shape[0], -1)) 
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # LSTM layer
        if actor_hidden_state is None:
            actor_hidden_state = (jnp.zeros(actor_mean.shape), jnp.zeros(actor_mean.shape))
        ScanLSTM = nn.vmap(  
            nn.OptimizedLSTMCell,  # processes a whole batch
            in_axes=0,
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )
        actor_lstm = ScanLSTM(self.hidden_dim)
        actor_hidden_state, actor_mean = actor_lstm(actor_hidden_state, actor_mean)
        # apply layer norm
        actor_mean = nn.LayerNorm()(actor_mean)

        if self.model_other_agent:
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(actor_mean)  # batch size by 128
            other_actor_mean = activation(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = activation(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_pi = distrax.Categorical(logits=other_actor_mean)
        
        elif self.e3t_baseline:  # condition on other agent action prediction and our own obs, train separate network
            b_size, num_time = previous_state_action['action'].shape[0], previous_state_action['action'].shape[1]
            past_graphs = jax.vmap(make_whole_graph)(previous_state_action['obs'].reshape((b_size, num_time, -1)))
            
            batch_graph = nn.vmap(
                GATLayer,
                in_axes=1, out_axes=1,
                variable_axes={'params': 1},
                split_rngs={'params': True}
            )

            other_graph_vals = batch_graph(c_out=16, num_heads=2, concat_heads=True)(*past_graphs)
            other_graph_vals = other_graph_vals.reshape((b_size, num_time, -1)) 
            
            embeddings = nn.Embed(num_embeddings=self.action_dim, features=self.hidden_dim // 2)(previous_state_action['action'].astype(jnp.int32))
            embeddings = embeddings.reshape((b_size, num_time, -1))  # remove extraneous dimension
            other_actor_mean = jnp.concatenate([other_graph_vals, embeddings], axis=-1)  # concatenate along feature dimension

            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            
            other_actor_mean = other_actor_mean.reshape((b_size, -1))  # concatenate 5 timepoints
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.tanh(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)

            other_pi = distrax.Categorical(logits=other_actor_mean)
            actor_mean = jnp.concatenate([actor_mean, other_actor_mean], axis=-1)

        # action layer 1
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)  # batch size by 128
        actor_mean = activation(actor_mean)
        # action layer 2
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # action output
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        if not (self.model_other_agent or self.e3t_baseline):
            other_pi = distrax.Categorical(logits=jnp.zeros_like(actor_mean))


        # feature detection 1
        critic = GATLayer(c_out=32, num_heads=4, concat_heads=True)(graph_vals, graph_edges)
        critic = critic.reshape((x.shape[0], -1))
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # LSTM layer
        if critic_hidden_state is None:
            critic_hidden_state = (jnp.zeros(critic.shape), jnp.zeros(critic.shape))
        ScanLSTM = nn.vmap(  
            nn.OptimizedLSTMCell,  # processes a whole batch
            in_axes=0,
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )
        critic_lstm = ScanLSTM(self.hidden_dim)
        critic_hidden_state, critic = critic_lstm(critic_hidden_state, critic)
        # apply layer norm
        critic = nn.LayerNorm()(critic)

        # critic layer 1
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic layer 2
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic output
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_hidden_state, critic_hidden_state, other_pi
    

class MlpLstmActor(nn.Module):
    action_dim: Sequence[int]
    observation_shape: tuple
    activation: str = "tanh"
    model_other_agent: bool = False
    e3t_baseline: bool = False
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, actor_hidden_state=None, critic_hidden_state=None, previous_state_action=None, observation_shape=None):
        '''
        Takes flattened observation as input (h * w * c)
        '''
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if observation_shape is None:
            observation_shape = self.observation_shape
        
        
        if len(x.shape) == 1:  # need to add batch dim if it's not already there
            x = x[None]
        
        # feature detection 1
        actor_mean = x.reshape((x.shape[0], -1))
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # LSTM layer
        if actor_hidden_state is None:
            actor_hidden_state = (jnp.zeros(actor_mean.shape), jnp.zeros(actor_mean.shape))
        ScanLSTM = nn.vmap(  
            nn.OptimizedLSTMCell,  # processes a whole batch
            in_axes=0,
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )
        actor_lstm = ScanLSTM(self.hidden_dim)
        actor_hidden_state, actor_mean = actor_lstm(actor_hidden_state, actor_mean)
        # apply layer norm
        actor_mean = nn.LayerNorm()(actor_mean)
        
        if self.model_other_agent:
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(actor_mean)  # batch size by 128
            other_actor_mean = activation(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = activation(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_pi = distrax.Categorical(logits=other_actor_mean)

        elif self.e3t_baseline:  # condition on other agent action prediction and our own obs, train separate network
            b_size, num_time = previous_state_action['action'].shape[0], previous_state_action['action'].shape[1]

            other_graph_vals = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(previous_state_action['obs'].reshape((b_size, num_time, -1)))
            other_graph_vals = activation(other_graph_vals)
            
            embeddings = nn.Embed(num_embeddings=self.action_dim, features=self.hidden_dim)(previous_state_action['action'].astype(jnp.int32))
            embeddings = embeddings.reshape((b_size, num_time, -1))  # remove extraneous dimension
            other_actor_mean = jnp.concatenate([other_graph_vals, embeddings], axis=-1)  # concatenate along feature dimension

            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            
            other_actor_mean = other_actor_mean.reshape((b_size, -1))  # concatenate 5 timepoints
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.tanh(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)

            other_pi = distrax.Categorical(logits=other_actor_mean)
            actor_mean = jnp.concatenate([actor_mean, other_actor_mean], axis=-1)

        # action layer 1
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)  # batch size by 128
        actor_mean = activation(actor_mean)
        # action layer 2
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # action output
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        if not (self.model_other_agent or self.e3t_baseline):
            other_pi = distrax.Categorical(logits=jnp.zeros_like(actor_mean))


        # feature detection 1
        critic = x.reshape((x.shape[0], -1))
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # LSTM layer
        if critic_hidden_state is None:
            critic_hidden_state = (jnp.zeros(critic.shape), jnp.zeros(critic.shape))
        ScanLSTM = nn.vmap(  
            nn.OptimizedLSTMCell,  # processes a whole batch
            in_axes=0,
            out_axes=0,
            variable_axes={'params': None},
            split_rngs={'params': False}
        )
        critic_lstm = ScanLSTM(self.hidden_dim)
        critic_hidden_state, critic = critic_lstm(critic_hidden_state, critic)
        # apply layer norm
        critic = nn.LayerNorm()(critic)

        # critic layer 1
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic layer 2
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic output
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_hidden_state, critic_hidden_state, other_pi

class MlpActor(nn.Module):
    action_dim: Sequence[int]
    observation_shape: tuple
    activation: str = "tanh"
    model_other_agent: bool = False
    e3t_baseline: bool = False
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, actor_hidden_state=None, critic_hidden_state=None, previous_state_action=None, observation_shape=None):
        '''
        Takes flattened observation as input (h * w * c)
        '''
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if observation_shape is None:
            observation_shape = self.observation_shape
        
        
        if len(x.shape) == 1:  # need to add batch dim if it's not already there
            x = x[None]
        
        # feature detection 1
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = actor_mean.reshape((x.shape[0], -1)) 

        if self.model_other_agent:
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(actor_mean)  # batch size by 128
            other_actor_mean = activation(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = activation(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_pi = distrax.Categorical(logits=other_actor_mean)

        elif self.e3t_baseline:  # condition on other agent action prediction and our own obs, train separate network
            b_size, num_time = previous_state_action['action'].shape[0], previous_state_action['action'].shape[1]

            other_graph_vals = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(previous_state_action['obs'].reshape((b_size, num_time, -1)))
            other_graph_vals = activation(other_graph_vals)
            
            embeddings = nn.Embed(num_embeddings=self.action_dim, features=self.hidden_dim)(previous_state_action['action'].astype(jnp.int32))
            embeddings = embeddings.reshape((b_size, num_time, -1))  # remove extraneous dimension
            other_actor_mean = jnp.concatenate([other_graph_vals, embeddings], axis=-1)  # concatenate along feature dimension

            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            
            other_actor_mean = other_actor_mean.reshape((b_size, -1))  # concatenate 5 timepoints
            # action layer 1
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)  # batch size by 128
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action layer 2
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)
            # action output
            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.leaky_relu(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.hidden_dim // 2, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)
            other_actor_mean = nn.activation.tanh(other_actor_mean)

            other_actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(other_actor_mean)

            other_pi = distrax.Categorical(logits=other_actor_mean)
            actor_mean = jnp.concatenate([actor_mean, other_actor_mean], axis=-1)  # condition ego policy on other agent action prediction


        # action layer 1
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)  # batch size by 128
        actor_mean = activation(actor_mean)
        # action layer 2
        actor_mean = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        # action output
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)
        if not (self.model_other_agent or self.e3t_baseline):
            other_pi = distrax.Categorical(logits=jnp.zeros_like(actor_mean))


        # feature detection 1
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = critic.reshape((x.shape[0], -1))
        # critic layer 1
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic layer 2
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        # critic output
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_hidden_state, critic_hidden_state, other_pi