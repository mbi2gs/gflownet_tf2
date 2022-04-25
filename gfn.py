import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten
import tensorflow_probability as tfp
tfd = tfp.distributions

from env import RewardEnvironment


class GFNAgent():
    """Example Generative Flow Network as described in:
    https://arxiv.org/abs/2106.04399 and
    https://arxiv.org/abs/2201.13259
    """
    def __init__(self,
                 env_dim=2,
                 env_len=8,
                 env_r0=0.01,
                 name='',
                 n_hidden=32,
                 gamma=0.5,
                 epochs=100,
                 lr=0.005,
                ):
        """Initialize GFlowNet agent.
        :param env_dim: (int) Number of dimensions in the reward environment
        :param env_len: (int) Length of each dimension in the environment
        :param env_r0: (float) r0 value in the environment
        :param name: (str) Agent name
        :param n_hidden: (int) Number of nodes in hidden layer of neural network
        :param gamma: (float) Mixture proportion when sampling from forward policy,
                      and mixing with uniform distribution
        :param epochs: (int) Number of epochs to complete during training cycle
        :param lr: (float) Learning rate
        :return: (None) Initialized class object
        """
        assert env_dim > 1
        assert env_len > 2
        assert 0 <= gamma <= 1
        self.name = name
        self.dim = env_dim
        self.env_len = env_len
        self.max_trajectory_len = env_len * env_dim
        self.action_space = env_dim + 1
        self.stop_action = self.action_space - 1
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.epochs = epochs
        self.env = RewardEnvironment(dim=env_dim, H=env_len, r0=env_r0)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=10000,
            decay_rate=0.8
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.data = {'positions':None, 'actions':None, 'rewards':None}
        self.clear_eval_data()
        self.get_model()


    def get_model(self):
        """Initialize neural network using TensorFlow2 functional API.
        :return: (None)
        """
        # Accept one-hot encoded states as input
        input_ = Input(shape=(self.dim,self.env_len), name='input')
        flatten_1 = Flatten()(input_)
        dense_1 = Dense(
            units=self.n_hidden,
            activation='relu',
            kernel_regularizer='l2',
            name='dense_1'
        )(flatten_1)
        dense_2 = Dense(
            units=self.n_hidden,
            activation='relu',
            kernel_regularizer='l2',
            name='dense_2'
        )(dense_1)
        # Output log probabilities using log_softmax activation
        fpm = Dense(units=self.action_space, activation='log_softmax', name='forward_policy')(dense_2)
        bpm = Dense(units=self.dim, activation='log_softmax', name='backward_policy')(dense_2)
        # Z0 is a single learned value
        self.z0 = tf.Variable(0., name='z0', trainable=True)
        # Model output will be a list of tensors for both forward and backward
        self.model = Model(input_, [fpm, bpm])
        # We'll be using the uniform distribution to add
        # more exploration to our forward policy
        self.unif = tfd.Uniform(low=[0]*self.action_space, high=[1]*self.action_space)


    def mask_forward_actions(self, batch_of_positions):
        """Build boolean mask with zeros over coordinates at the edge of environment.
        :param batch_of_positions: (nd.array) Array of coordinates
        :return: (nd.array) Mask over coordinates at the edge of environment with same
                 shape == batch_of_positions.shape
        """
        batch_size = batch_of_positions.shape[0]
        # Check that we're not up against the edge of the environment
        action_mask = batch_of_positions < (self.env_len - 1)
        # The "stop" action is last and always allowed, so we append a 1 at the end)
        stop_column = np.ones((batch_size, 1))
        return np.append(action_mask, stop_column, axis=1)


    def mask_and_norm_forward_actions(self, batch_positions, batch_forward_probs):
        """Remove actions that would move outside the environment, and re-normalize
        probabilities so that they sum to one.
        :param batch_positions: (nd.array) Array of coordinates
        :param batch_forward_probs: (nd.array) Array of probabilites over actions
        :return: (nd.array) Masked and re-normalized probabilities
        """
        mask = self.mask_forward_actions(batch_positions)
        masked_actions = mask * batch_forward_probs.numpy()
        # Normalize masked probabilities so that they again sum to 1
        normalized_actions = masked_actions / np.sum(masked_actions, axis=1, keepdims=True)
        return normalized_actions


    def sample_trajectories(self, batch_size=3, explore=False):
        """Sample `batch_size` trajectories using the current policy.
        :param batch_size: (int) Number of trajectories to sample
        :param explore: (bool) If True, mix policy with uniform distribution
                        to encourage exploration.
        :return: (tuple of nd.array) (trajectories, one_hot_actions, rewards)
        """
        # Start at the origin
        still_sampling = [True]*batch_size
        positions = np.zeros((batch_size, self.dim), dtype='int32')
        trajectories = [positions.copy()]
        one_hot_actions = []
        batch_rewards = np.zeros((batch_size,))
        for step in range(self.max_trajectory_len-1):
            # Convert positions to one-hot encoding
            one_hot_position = tf.one_hot(positions, self.env_len, axis=-1)
            # Use forward policy to get log probabilities over actions
            model_fwrd_logits = self.model.predict(one_hot_position)[0]
            model_fwrd_probs = tf.math.exp(model_fwrd_logits)
            if explore:
                # Mix with uniform distribution to encourage exploration
                unif = self.unif.sample(sample_shape=model_fwrd_probs.shape[0])
                model_fwrd_probs = self.gamma*unif + (1-self.gamma)*model_fwrd_probs
            # Don't select impossible actions (like moving out of the environment)
            normalized_actions = self.mask_and_norm_forward_actions(positions, model_fwrd_probs)
            # Select actions randomly, proportionally to input probabilities
            actions = tfd.Categorical(probs=normalized_actions).sample()
            actions_one_hot = tf.one_hot(actions, self.action_space).numpy()
            # Update positions based on selected actions
            for i, act_i in enumerate(actions):
                if act_i == (self.action_space - 1) and still_sampling[i]:
                    still_sampling[i] = False
                    batch_rewards[i] = self.env.get_reward(positions[i,:])
                elif not still_sampling[i]:
                    positions[i,:] = -1
                    actions_one_hot[i,:] = 0
                else:
                    positions[i, act_i] += 1
            trajectories.append(positions.copy())
            one_hot_actions.append(actions_one_hot)
        return np.stack(trajectories, axis=1), np.stack(one_hot_actions, axis=1), batch_rewards


    def mask_and_norm_backward_actions(self, position, backward_probs):
        """Set probabilities to zero where action to lead out of the environment.
        :param position: (nd.array) Coordinate in the environment
        :param backward_probs: (nd.array) Array of probabilities
        :return: (nd.array) Masked and re-normalized probabilities
        """
        assert isinstance(position, np.ndarray)
        mask = (position > 0).astype(int)
        masked_actions = position * backward_probs.numpy()
        # Normalize masked probabilities so that they again sum to 1
        normalized_actions = masked_actions / np.sum(masked_actions, axis=1, keepdims=True)
        return normalized_actions


    def back_sample_trajectory(self, position):
        """Follow current backward policy from a position back to the origin.
        Returns them in in "forward order" such that origin is first.
        :param position: (nd.array) Coordinate in the environment
        :return: (tuple of nd.array) (positions, actions)
        """
        # Trace a path back to the origin from a given position
        assert isinstance(position, np.ndarray)
        one_hot_position = tf.one_hot(np.expand_dims(position, 0), self.env_len, axis=-1)
        positions = [one_hot_position]
        actions = [tf.one_hot(self.action_space-1, self.action_space).numpy()]
        still_tracing = True
        if np.all(position == [0,0]):
            still_tracing = False
        cur_pos = position.copy()
        while still_tracing:
            # Use backward policy to get log probabilities over non-termination-actions
            model_back_logits = self.model(one_hot_position)[1]
            model_back_probs = tf.math.exp(model_back_logits)
            # Don't select impossible actions (like moving out of the environment)
            normalized_actions = self.mask_and_norm_backward_actions(cur_pos, model_back_probs)
            # Select most likely action
            action = np.argmax(normalized_actions)
            prob = np.max(normalized_actions)
            action_one_hot = tf.one_hot(action, self.action_space).numpy()
            # Update position based on selected action
            cur_pos[action] -= 1
            # Convert position to one-hot encoding
            one_hot_position = tf.one_hot(np.expand_dims(cur_pos, 0), self.env_len, axis=-1)
            # Stop tracing if at origin
            if np.all(cur_pos == [0,0]):
                still_tracing = False

            positions.append(one_hot_position.numpy().copy())
            actions.append(action_one_hot.copy())
        return (
            np.flip(np.concatenate(positions, axis=0), axis=0),
            np.flip(np.stack(actions, axis=0), axis=0)
        )


    def get_last_position(self, trajectory):
        """Identify the termination coordinates for a trajectory.
        :param trajectory: (nd.array) Array of coordinates/positions
        :return: (nd.array) Last position
        """
        assert len(trajectory.shape) == 2
        mask = trajectory != -1
        traj_no_pad = trajectory[mask[:,0],...]
        last_position = traj_no_pad[-1,:]
        return last_position


    def sample(self, num_to_sample, explore=True, evaluate=False):
        """Sample trajectories using the current policy and save to
        `self.data` or `self.eval_data`.
        :param num_to_sample: (int) Number of samples to collect
        :param explore: (bool) If True, uniform distribution is mixed with current policy,
                        and output is saved to `self.data`.
        :param evaluate: (bool) If False, trajectories are de-duplicated,
                        and output is saved to `self.eval_data`
        :return: (None) Data saved internally
        """
        assert num_to_sample > 0
        trajectories, actions, rewards = self.sample_trajectories(
            batch_size=num_to_sample,
            explore=explore
        )
        positions = np.stack([self.get_last_position(x) for x in trajectories], axis=0)
        if not evaluate:
            if self.data['positions'] is not None:
                # (batch, len_trajectory, env dimensions)
                self.data['trajectories'] = np.append(self.data['trajectories'], trajectories, axis=0)
                # (batch, env dimensions)
                self.data['positions'] = np.append(self.data['positions'], positions, axis=0)
                # (batch, len_trajectory-1, action dimensions)
                self.data['actions'] = np.append(self.data['actions'], actions, axis=0)
                # (batch,)
                self.data['rewards'] = np.append(self.data['rewards'], rewards, axis=0)
            else:
                self.data['trajectories'] = trajectories
                self.data['positions'] = positions
                self.data['actions'] = actions
                self.data['rewards'] = rewards
            # Ensure that training data do not contain duplicates
            # (simply to make training faster)
            u_positions, u_indices = np.unique(positions, axis=0, return_index=True)
            self.data['trajectories'] = self.data['trajectories'][u_indices]
            self.data['positions'] = u_positions
            self.data['actions'] = self.data['actions'][u_indices]
            self.data['rewards'] = self.data['rewards'][u_indices]
        else:
            # For evaluating frequencies we have to keep duplicates
            self.eval_data['trajectories'] = trajectories
            self.eval_data['positions'] = positions
            self.eval_data['actions'] = actions
            self.eval_data['rewards'] = rewards


    def clear_eval_data(self):
        """Refresh self.eval_data dictionary."""
        self.eval_data = {'positions':None, 'actions':None, 'rewards':None}


    def train_gen(self, batch_size=10):
        """Generator object that feeds shuffled samples from `self.data` to training loop.
        :return: (tuple of ndarrays) (positions, rewards)
        """
        data_len = self.data['rewards'].shape[0]
        assert data_len > 0
        iterations = int(data_len // batch_size) + 1
        shuffle = np.random.choice(data_len, size=data_len, replace=False)
        for i in range(iterations):
            # Pick a random batch of training data
            samples = shuffle[i*batch_size:(i+1)*batch_size]
            sample_positions = self.data['positions'][samples]
            sample_rewards = tf.convert_to_tensor(self.data['rewards'][samples], dtype='float32')
            yield (sample_positions, sample_rewards)


    def trajectory_balance_loss(self, batch):
        """Calculate Trajectory Balance Loss function as described in
        https://arxiv.org/abs/2201.13259.
        I added an additional piece to the loss function to penalize
        actions that would extend outside the environment.
        :param batch: (tuple of ndarrays) Output from self.train_gen() (positions, rewards)
        :return: (list) Loss function as tensor for each value in batch
        """
        positions, rewards = batch
        losses = []
        for i, position in enumerate(positions):
            reward = rewards[i]
            # Sample a trajectory for the given position using backward policy
            trajectory, back_actions = self.back_sample_trajectory(position)
            # Generate policy predictions for each position in trajectory
            tf_traj = tf.convert_to_tensor(trajectory[:,...], dtype='float32')
            forward_policy, back_policy = self.model(tf_traj)
            # Use "back_actions" to select corresponding forward probabilities
            forward_probs = tf.reduce_sum(
                tf.multiply(forward_policy, back_actions),
                axis=1
            )
            # Get backward probabilities for the sampled trajectory (ignore origin)
            backward_probs = tf.reduce_sum(
                tf.multiply(back_policy[1:,:], back_actions[:-1,:self.dim]),
                axis=1
            )
            # Add a constant backward probability for transitioning from the termination state
            backward_probs = tf.concat([backward_probs, [0]], axis=0)
            # take log of product of probabilities (i.e. sum of log probabilities)
            sum_forward = tf.reduce_sum(forward_probs)
            sum_backward = tf.reduce_sum(backward_probs)
            # Calculate trajectory balance loss function and add to batch loss
            numerator = self.z0 + sum_forward
            denominator = tf.math.log(reward) + sum_backward
            tb_loss = tf.math.pow(numerator - denominator, 2)

            # Penalize any probabilities that extend beyond the environment
            # This part is not from the publication
            fwrd_edges = tf.cast(
                np.argmax(trajectory, axis=2) == (self.env_len-1),
                dtype='float32'
            )
            back_edges = tf.cast(
                np.argmax(trajectory, axis=2) == 0,
                dtype='float32'
            )
            fedge_probs = tf.math.multiply(
                tf.math.exp(forward_policy[:,:self.dim]),
                fwrd_edges
            )
            bedge_probs = tf.math.multiply(
                tf.math.exp(back_policy[:,:self.dim]),
                back_edges
            )[1:,:] # Ignore backward policy for the origin
            fedge_loss = tf.reduce_sum(fedge_probs)
            bedge_loss = tf.reduce_sum(bedge_probs)
            combined_loss = tf.math.add(tb_loss, tf.math.add(fedge_loss, bedge_loss))
            losses.append(combined_loss)
        return losses


    def grad(self, batch):
        """Calculate gradients based on loss function values. Notice the z0 value is
        also considered during training.
        :param batch: (tuple of ndarrays) Output from self.train_gen() (positions, rewards)
        :return: (tuple) (loss, gradients)
        """
        with tf.GradientTape() as tape:
            loss = self.trajectory_balance_loss(batch)
            grads = tape.gradient(loss, self.model.trainable_variables + [self.z0])
        return loss, grads


    def train(self, verbose=True):
        """Run a training loop of `length self.epochs`.
        At the end of each epoch, save weights if loss is better than any previous epoch.
        At the end of training, read in the best weights.
        :param verbose: (bool) Print additional messages while training
        :return: (None) Updated model parameters
        """
        if verbose: print('Start training...')
        # Keep track of loss during training
        train_loss_results = []
        best_epoch_loss = 10**10
        model_weights_path = './checkpoints/gfn_checkpoint'
        for epoch in range(self.epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            sampler = self.train_gen()
            # Iterate through shuffled batches of deduplicated training data
            for batch in sampler:
                loss_values, gradients = self.grad(batch)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables + [self.z0])
                )
                losses = []
                for sample in loss_values:
                    losses.append(sample.numpy())
                epoch_loss_avg(np.mean(losses))
            # If current loss is better than any previous, save weights
            if epoch_loss_avg.result() < best_epoch_loss:
                self.model.save_weights(model_weights_path)
                best_epoch_loss = epoch_loss_avg.result()

            train_loss_results.append(epoch_loss_avg.result())
            if verbose and epoch % 9 == 0: print(f'Epoch: {epoch} Loss: {epoch_loss_avg.result()}')
        # Load best weights
        self.model.load_weights(model_weights_path)


    def compare_env_to_model_policy(self, sample_size=2000, plot=True):
        """Compare probability distribution over generated trajectories
        (estimated empirically) to reward distribution in environment.
        Compare using L1 error.
        :param sample_size: (int) Number of samples used to estimate probability of
                            trajectories terminating in each position of environment.
        :param plot: (bool) Plot first two dimensions of empirical distribution
        :return: (float) L1 error
        """
        # Start data set from a clean, on-policy slate
        self.clear_eval_data()
        self.sample(sample_size, explore=False, evaluate=True)
        env_prob = self.env.env_prob
        agent_prob = np.zeros(env_prob.shape)
        # Count number of trajectories that end in each position,
        # and normalize by the total
        for i_pos in range(self.eval_data['positions'].shape[0]):
            last_position = self.eval_data['positions'][i_pos,...]
            agent_prob[tuple(last_position)] += 1
        agent_prob = agent_prob / np.sum(agent_prob)

        if plot:
            top_slice = tuple([slice(0,self.env_len),slice(0,self.env_len)] + [0]*(self.dim - 2))
            plt.imshow(agent_prob[top_slice], origin='lower');
        
        l1_error = np.sum(np.abs(agent_prob - env_prob))
        return l1_error


    def count_modes(self):
        """Count the number of modes sampled in `self.data`. Modes being
        local maxima.
        :return: (tuple) (num_unique_modes, num_unique_terminal_positions)
        """
        assert self.data['rewards'] is not None
        # Count the number of total positions sampled in the data set
        all_positions_list = []
        # Count the number of modes (positions of maximum reward) sampled in the data set
        modes_list = []
        for i_pos in range(self.data['positions'].shape[0]):
            last_position = self.data['positions'][i_pos,...]
            all_positions_list.append(str(last_position))
            if self.env.get_reward(last_position) == self.env.reward.max():
                modes_list.append(str(last_position))
        return len(set(modes_list)), len(set(all_positions_list))


    def plot_policy_2d(self):
        """Plot forward and backward policies.
        :return: (None) Matplotlib figure
        """
        # Generate grid coordinates
        top_slice = tuple([slice(0,self.env.length),slice(0,self.env.length)] + [0]*(self.dim - 2))
        coordinates = []
        for coord, i in np.ndenumerate(self.env.reward[top_slice]):
            coordinates.append(coord)
        coords = np.array(coordinates)
        one_hot_position = tf.one_hot(coords, self.env_len, axis=-1)
        # Use forward policy to get probabilities over actions
        frwd_logits, back_logits = self.model.predict(one_hot_position)
        model_fwrd_prob = tf.math.exp(frwd_logits).numpy()
        model_back_prob = tf.math.exp(back_logits).numpy()
        fig, axes = plt.subplots(ncols=2, figsize=(10,5))
        # Arrows for forward probabilities
        for i in range(coords.shape[0]):
            for act in [0,1]:
                x_change = 0
                y_change = model_fwrd_prob[i,act]
                if act == 1:
                    x_change = model_fwrd_prob[i,act]
                    y_change = 0
                axes[0].arrow(
                    coords[i,1],
                    coords[i,0],
                    x_change,
                    y_change,
                    width=0.04,
                    head_width=0.1,
                    fc='black',
                    ec='black'
                )
        # Arrows for backward probabilities
        for i in range(coords.shape[0]):
            for act in [0,1]:
                x_change = 0
                y_change = -model_back_prob[i,act]
                if act == 1:
                    x_change = -model_back_prob[i,act]
                    y_change = 0
                axes[1].arrow(
                    coords[i,1],
                    coords[i,0],
                    x_change,
                    y_change,
                    width=0.04,
                    head_width=0.1,
                    fc='black',
                    ec='black'
                )
        # Stop probabilities marked with red octagons (forward only)
        axes[0].scatter(
            coords[:,1],
            coords[:,0],
            s=model_fwrd_prob[:,2]*200,
            marker='8',
            color='red'
        )
        # Titles
        axes[0].set_title('Forward policy')
        axes[1].set_title('Backward policy')
        plt.show()


    def plot_sampled_data_2d(self):
        """Plot positions and associated rewards found in `self.data`.
        :return: (None) Matplotlib figure
        """
        assert self.dim == 2
        fig, ax = plt.subplots(nrows=1, figsize=(5,5))
        all_positions = self.data['positions']
        ax.scatter(
            all_positions[:,1],
            all_positions[:,0],
            marker='x',
            color='red',
            s=self.data['rewards']*50
        )
        plt.show()
