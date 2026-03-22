
import random
from itertools import combinations_with_replacement, combinations

import numpy as np

import torch
import torch.nn as nn


from scipy.linalg import expm
from scipy.optimize import minimize_scalar


# Two-body operators as RL action for CSE
class CSE_action:
    # generate indices of two-body operators, Gamma_ij^kl
    @staticmethod
    def Get_indices(n_orbit: int) -> list[tuple]:
        '''
        Args:
            - n_orbit (int): number of orbitals
            - full_version (bool): True for including some redundant terms
        '''
        # even index: for spin-up orbitals (=> site)
        spin_up_indices = [i for i in range(0, n_orbit, 2)]
        # odd index: for spin-down orbitals (=> site)
        spin_down_indices = [i for i in range(1, n_orbit, 2)]
        # Generate all unique (i, j) pairs for spin-up where i != j
        spin_up_pairs = [pair for pair in combinations(spin_up_indices, 2)]
        # Generate all unique (i, j) pairs for spin-down where i != j
        spin_down_pairs = [pair for pair in combinations(spin_down_indices, 2)]
        # one from spin-up, one from spin-down
        spin_up_down_pairs = list(combinations_with_replacement(spin_up_indices + spin_down_indices, 2))
        valid_spin_up_down_pairs = [pair for pair in spin_up_down_pairs 
                                    if (pair[0] in spin_up_indices and pair[1] in spin_down_indices) \
                                    or (pair[0] in spin_down_indices and pair[1] in spin_up_indices)]
        # Generate all valid (i, j, k, l) combinations for spin-up and spin-down
        valid_combinations = []
        # simplified version
        for ij in spin_up_pairs:
            for kl in spin_up_pairs:
                valid_combinations.append((ij[0], ij[1], kl[0], kl[1]))

        for ij in spin_down_pairs:
            for kl in spin_down_pairs:
                valid_combinations.append((ij[0], ij[1], kl[0], kl[1]))

        for ij in valid_spin_up_down_pairs:
            for kl in valid_spin_up_down_pairs:
                valid_combinations.append((ij[0], ij[1], kl[0], kl[1]))
        
        return valid_combinations


    # list of two-body operators 
    @staticmethod
    def Get_matrices_Hermitian(n_orbit: int, index_list: list[tuple],
                               Hermitian_reduction: bool = True) -> list[np.ndarray]:
        '''
        If use Hermitian reduction: 
            - G[i,j,k,l] + G[l,k,j,i] will be used in exponential
            - G[i,j,k,l] will be used, and duplicated G[l,k,j,i] will be dropped
        '''      
        # Pauli matrix
        sigma_x = np.array([[0, 1],  [1, 0]] , dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]] , dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]] , dtype=complex)
        iden    = np.array([[1, 0],  [0, 1]] , dtype=complex)
        # Qubit tomography is used here, annihilation operator
        def anni(pos: int, N: int) -> np.ndarray:
            '''           
            c_n ~ sigma_n^x + i * sigma_n^y
            '''
            mat = 1
            for i in range(N):
                if i == pos :
                    mat = np.kron(mat, (sigma_x+1j*sigma_y)/2)
                else:
                    mat = np.kron(mat, iden)
            return mat

        # creation operator
        def crea(pos: int, N: int) -> np.ndarray:
            '''
            c_n ~ sigma_n^x - i * sigma_n^y
            '''
            mat = 1
            for i in range(N):
                if i==pos :
                    mat = np.kron(mat, (sigma_x-1j*sigma_y)/2)
                else:
                    mat = np.kron(mat, iden)
            return mat

        # 2-RDM operator Gamma_(ij)^(kl) at index (i, j, k, l)
        def rdmtwo(i: int, j: int, k:int , l: int, N: int) -> np.ndarray:
            '''
            2-RDM operator Gamma_(i,j)^(k,l)

            Args:
                - N (int): total number of sites
            '''
            # matrix production (from right to left)
            tomo = anni(k, N)
            tomo = np.matmul(anni(l, N), tomo)
            tomo = np.matmul(crea(j, N), tomo)
            tomo = np.matmul(crea(i, N), tomo)
            return tomo

        # list
        mat_list = []
        if Hermitian_reduction:
            dup_dict = {}
            for idx in index_list:
                idx_herm = (idx[2], idx[3], idx[0], idx[1])
                if idx not in dup_dict and idx_herm not in dup_dict:
                    mat_list.append(rdmtwo(*idx, n_orbit))
                    dup_dict[idx] = True
                    dup_dict[idx_herm] = True
        else:
            dup_dict = {}
            for idx in index_list:
                if idx not in dup_dict:
                    mat_list.append(rdmtwo(*idx, n_orbit))
                    dup_dict[idx] = True
                idx_herm = (idx[2], idx[3], idx[0], idx[1])
                if idx_herm not in dup_dict:
                    mat_list.append(rdmtwo(*idx_herm, n_orbit))
                    dup_dict[idx_herm] = True
            del dup_dict
        return mat_list


    # initial wavefunction 
    @staticmethod
    def Initial_ortho_wf(num_orbitals: int = 4, num_ensemble: int = 3,
                         num_spin_up: int = 1, num_spin_down = 1) -> np.ndarray:
        '''
        num_orbitals: total number of spin-orbitals

        num_ensemble: number of excited states (K value)

        Return:
            - size (N, K) matrix
        '''
        valid_indices = []
        num_occupied = num_spin_up + num_spin_down
        for state in combinations(range(num_orbitals), num_occupied):
            binary_state = [0] * num_orbitals
            for idx in state:
                binary_state[idx] = 1
            # number of spin-up (even) and spin-down electrons
            up_count = sum(binary_state[i] for i in range(0, num_orbitals, 2))
            down_count = sum(binary_state[i] for i in range(1, num_orbitals, 2))

            # ensure N_1 spin-up and N_2 spin-down electrons
            if up_count == num_spin_up and down_count == num_spin_down: 
                index = int("".join(map(str, binary_state)), 2)
                valid_indices.append(index)

        wavefunction = np.zeros((2**num_orbitals, num_ensemble), dtype=complex)
        # H2 molecule 
        for k in range(num_ensemble):
            wavefunction[valid_indices[k], k] = 1.0
        return wavefunction



# Excited states via purified ensemble
class CSE_purified:
    '''
    Args:
        - initial_wf (array): size (N) reference wavefunction |psi_0> 
        - H_mat (array): size (N, N) Hamiltonian matrix
        - action_list (list[array]): size [(N, N)][num_action] discrete actions
        - action_commute_list (list[array]): commutator of actions and Hamiltonian [O_i, H_mat]
        - residual_termination (float): terminate when CSE residual is small enough (approach correct state)
        - cse_weight_factor: weight factor in CSE reward [Eq.(8) in original work]
    '''
    def __init__(self, weight_vec: np.ndarray, initial_wf: np.ndarray, 
                 H_mat: np.ndarray, action_list: list[np.ndarray],
                 eta_factor: float = 0.1,
                 residual_termination: float = 1e-20, 
                 cse_weight_factor: float = 1.0):
        # system Hamiltonian and actions
        self.H_mat = H_mat
        self.action_list = action_list
        self.weight_vec = np.asarray(weight_vec, dtype=float)       
        # commutator of action and Hamiltonian
        self.commute_list = [np.dot(Ax, self.H_mat) - np.dot(self.H_mat, Ax)
                             for Ax in self.action_list]
        # initial RL state
        self.initial_wf = self._normalize(initial_wf).copy()  
        self.current_wf = self.initial_wf.copy()        
        self.initial_state = self.CSE_residual(self.current_wf)
        # RL parameters
        self.residual_termination = residual_termination
        self.cse_weight_factor = cse_weight_factor
        self.eta_factor = eta_factor
        ### 


    # reset RL state to initial one
    def Reset(self) -> np.ndarray:
        '''
        reset RL state to initial value
        '''
        self.current_wf = self.initial_wf.copy()
        return self.initial_state.copy()


    # expectation value of operator
    def _expect(self, wf: np.ndarray, H_mat: np.ndarray) -> float:
        '''
        Expectation of Hermitian operator <psi|H|psi>

        Args:
            - wf (array): size (N, K) matrix 
            - H_mat (array) size (N, N) Hermitian matrix
        '''
        result = np.einsum('xb,xy,yb->b', np.conj(wf), H_mat, wf)
        result = self.weight_vec * np.asarray(result, dtype=complex)
        return np.sum(result) 


    # normalize
    def _normalize(self, wf: np.ndarray) -> np.ndarray:
        '''
        normalize wavefunction, psi = psi/sqrt(|psi|)

        wf: size (N, K) matrix
        '''     
        norm = np.einsum('xb,xb->b', np.conj(wf), wf)    
        wf /= np.sqrt(norm[np.newaxis, :].real) 
        return wf
        

    # energy value (ensemble)
    def energy_value(self, wf: np.ndarray) -> np.ndarray:
        res = np.einsum('xb,xy,yb->b', np.conj(wf), self.H_mat, wf)
        res = np.asarray(res, dtype=complex)
        return res.real
        

    # residual of CSE
    def CSE_residual(self, wf: np.ndarray) -> np.ndarray:
        '''
        RL state: residual of CSE
            - R_i = <phi|[O_i, H]|phi>

        Return:
            - real-valued residual
        '''
        # value = <psi|[O_i, H]|psi> for all O_i in operator list
        residuals = []
        for item in self.commute_list:      
            residuals.append(self._expect(wf, item))
        residuals = np.array(residuals).real
        return np.asarray(residuals, dtype=float)
   

    # |phi_n> = exp(a * A) |phi_(n-1)>
    def apply_exp_operator(self, wf: np.ndarray, H_mat: np.ndarray,
                           index_action: int) -> np.ndarray:
        '''
        Unitary transformation
        '''
        # selected action
        action = self.action_list[index_action]
        ac_exp = 0.5 * (action - np.conj(action.T)) 
        # for scipy.optimize
        def objective(epsilon: float) -> float:
            operator = expm(epsilon * ac_exp)
            new_wf = np.dot(operator, wf)
            new_wf = self._normalize(new_wf)
            return self._expect(new_wf, H_mat).real
        # compute scalar part
        result = minimize_scalar(objective, method='brent')
        # Compute optimized wavefunction
        optimal_epsilon = result.x
        operator = expm(optimal_epsilon * ac_exp)
        optimal_wf = np.dot(operator, wf)
        optimal_wf = self._normalize(optimal_wf)        
        return optimal_wf


    # one incremental step of RL
    def Step(self, action: int) -> tuple[np.ndarray, np.ndarray, bool, float]:
        '''
        action: index of to-be-applied operator

        Given wavefunction [|psi(n-1)>], state [CSE residual], and to-be-applied operator A_i
        1. Apply A_i to |psi(n-1)> to get new wavefunction |psi(n)>
        2. Calculate energy expectation <psi(n)|H|psi(n)> as reward
        3. Update CSE residual, and use it as new state for next iteration

        Return:
            - next_state (after action has been applied)
            - reward
            - done (if terminate)
            - energy_value 
        '''
        # Apply the quantum operation and normalize new wavefunction
        self.current_wf = self.apply_exp_operator(
            self.current_wf, self.H_mat, action)
        # Residual of CSE (RL state) 
        # Terminate if residual is already small enough
        new_state = self.CSE_residual(self.current_wf)
        # Calculate reward
        energy_expect = self._expect(self.current_wf, self.H_mat).real
        residuals = new_state
        reward = (-energy_expect - self.cse_weight_factor * 
                  np.sum(np.abs(residuals)))
        # Done condition: terminate when residual is small enough
        # No need for small number of actions
        done = np.all(np.abs(residuals) < self.residual_termination)
        # energy value
        res_energy = self.energy_value(self.current_wf)
        return new_state, reward, done, res_energy



# Prioritized Replay buffer of RL for sampling 
class Replay_buffer:
    '''
    capacity: total number of samples in buffer

    alpha: scalar factor for calculating the sampling probability
    '''
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        # Small constant to prevent zero priority
        self.num_zero_const = 1e-6
        ###

    # add sample to buffer (Algorithm 1 in arXiv.1511.05952)
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        '''
        state: state (s_t) of current timestep

        action: action (a_t) of current timestep (represented as index)

        reward: reward (s_t) of current timestep

        next_state: new state after applied action 

        done: for termination (when approaching maximum step)
        '''
        # p_t = max_(i<t) p_i
        max_priority = self.priorities.max() if self.buffer else 1.0
        # add new samples to memory
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        # set priority p_t of new sample
        self.priorities[self.position] = max_priority
        # if size > capacity, delete oldest one
        self.position = (self.position + 1) % self.capacity
        return
        ###

    def sample(self, batch_size: int, beta: float) -> tuple[
        list[np.ndarray], np.ndarray, np.ndarray]:
        '''
        Shape of state/action array: (batch_size, state_dim/action_dim)

        Shape of other array: (batch_size)
        
        Return:
            - (state, action, reward, next_state, done)
            - weight: priority
            - indices: of random sample
        '''
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        # Sample indices and calculate importance sampling weights
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.buffer[idx] for idx in indices]
        states, discrete_actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states), np.array(discrete_actions),
                np.array(rewards), np.array(next_states),
                np.array(dones)), weights, indices

    def update_priorities(self, indices: list[int], td_errors: np.ndarray):
        '''
        indices: sampled index vector

        td_errors: length-1 array, for q-values
        '''
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error[0] + self.num_zero_const 
        return
    
    # if having enough samples to train
    def ready_to_train(self, batch_size: int) -> bool:
        return batch_size < len(self.buffer)
    
    # clear
    def clear(self) -> None:
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity)
        self.position = 0
        return
    ###
    



# Dueling double deep-Q network (D3QN)
class DQN_Network(nn.Module):
    '''
    state_dim: size of RL state descriptor (= residuals of CSE)

    num_actions: number of all possible RL actions (= number of 2-RDM operators)

    hidden_dim: hidden dimension of neural network
    '''
    def __init__(self, state_dim: int, num_actions: int, 
                 hidden_dim: int = 256, dtype: torch.dtype = torch.float,
                 device: torch.device = 'cpu'):
        super().__init__()
        self.device = device

        # shared layers of DDQN
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(dtype=dtype, device=self.device)

        # Value network
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        ).to(dtype=dtype, device=self.device)

        # Advantage network
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        ).to(dtype=dtype, device=self.device)

        # Initialize weights
        self.apply(self._init_weights)
        ###

    # orthognonal initialization
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.fill_(0.0)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)            
        # else:
            # raise NotImplementedError('Only implement for linear layers. ')
        return

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        '''
        state -> NN -> Q-value function Q(s,a) of each action
        '''
        # input -> shared network
        shared_features = self.shared_layers(state)
        # Dueling DQN computation
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        # Q-value = value-network + advantage-network
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values



# Action selection: epsilon-greedy algorithm
class DQN_Agent:
    '''
    state_dim: size of RL state descriptor 

    num_actions: number of all possible RL actions (discrete)

    buffer_capacity: capacity of reply buffer (for RL training)

    max_RL_step: upper limit of RL steps 

    About variable [epsilon]: controls whether to exploration or exploitation
        - epsilon-greedy algorithm: see [Algorithm 16 DDQN p.30] in arxiv.2411.18892
    '''
    def __init__(self, state_dim: int, num_actions: int,
                 max_step_RL: int = 5, dtype: torch.dtype = torch.float,
                 device: torch.device = 'cpu', print_step: int = 100,               
                 buffer_capacity: int = 5000):
        self.dtype = dtype; self.device = device
        # Initialize networks with improved architecture
        # policy network: state -> action
        self.policy_net = DQN_Network(
            state_dim, num_actions, dtype=dtype, device=self.device)
        # target network: used for training
        self.target_net = DQN_Network(
            state_dim, num_actions, dtype=dtype, device=self.device)
        # initialize target network 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=3e-4, weight_decay=1e-4)
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5)
        # Prioritized Experience Replay (change if necessary)
        self.memory_buffer = Replay_buffer(capacity=buffer_capacity, alpha=0.6)
        # number of environment actions
        self.num_actions = num_actions
        # Hyperparameters
        self.gamma_reward = 0.99 # weight factor in reward Q-function 
        # clip of gradient norm
        self.grad_norm_clip = 10
        # epsilon-greedy algorithm
        self.epsilon = 1.0 # decay expolartion rate
        self.epsilon_min = 0.01  # Lower minimum epsilon
        self.epsilon_decay = 0.995  # For demonstration 0.995 is used, which decay in ~900 episode
        # Update target network every several steps
        self.update_target_step = 10  
        self.print_step = print_step
        # For sampling from experience replay buffer
        self.beta = 0.4
        self.beta_increment = 0.001
        # Training metrics
        self.metrics = { 'episode_rewards': [], 'episode_steps': [],
                        'episode_residuals': [], 'episode_energy': [],
                        'episode_wavefunc': []}
        # upper limit of RL (time) step
        self.max_step = max_step_RL
        # loss function
        self.loss_func_body = torch.nn.functional.smooth_l1_loss
        ###

    # select action (index of operator) based on state
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        '''
        state: current RL state, (state_dim)

        Return:
            - action index (integer)
        '''
        # (state_dim) -> (1, state_dim)
        state = torch.as_tensor(state).unsqueeze(0).to(
            dtype=self.dtype, device=self.device)
        if training and random.random() < self.epsilon:
            action_index = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                # torch.max(input, dim=1)[1]: the index of maximum value 
                # torch.argmax(...) is the second return value of torch.max(...)
                action_index = q_values.max(dim=1)[1].item()
                # action_index = torch.argmax(q_values, dim=-1).item()
        return action_index


    # one training epoch
    def update_network(self, batch_size: int,  K_epochs: int = 1):
        '''
        K_epoch: number of epochs 

        batch_size: (...)
        '''
        for _ in range(K_epochs):
            # Sample with priorities and importance sampling
            # From stored experience replay buffer
            batch_data, weights, indices = self.memory_buffer.sample(batch_size, self.beta)
            states, actions, rewards, next_states, dones = batch_data
            # Convert numpy.array to torch.tensor and move to device (gpu)
            # (batch_size, state_dim)
            states = torch.as_tensor(states).to(
                dtype=self.dtype, device=self.device)
            # (batch_size) -> (batch_size, 1)
            actions = torch.as_tensor(actions).to(self.device).unsqueeze(-1)
            # (batch_size) -> (batch_size, 1)
            rewards = torch.as_tensor(rewards).unsqueeze(-1).to(
                dtype=self.dtype, device=self.device)
            # (batch_size, state_dim)
            next_states = torch.as_tensor(next_states).to(
                dtype=self.dtype, device=self.device)
            # (batch_size) -> (batch_size, 1)
            dones = torch.as_tensor(dones).unsqueeze(-1).to(
                dtype=self.dtype, device=self.device)
            # (batch_size) -> (batch_size, 1)
            weights = torch.as_tensor(weights).unsqueeze(-1).to(
                dtype=self.dtype, device=self.device)
            # Double DQN: policy + target network
            idx_gather = next_states.dim() - 1
            # For more details, see Eq.(6) of arXiv.1511.06581
            with torch.no_grad():
                # selected action, (batch_size, state_dim)
                next_q_values = self.policy_net(next_states) 
                # selected action, (batch_size) -> (batch_size, 1)
                next_actions = torch.argmax(next_q_values, dim=-1).unsqueeze(-1)
                # target_Q = Q(S_(t+1), action) but on target network
                next_q_values = self.target_net(next_states)
                next_q_values = next_q_values.gather(idx_gather, next_actions)
            # Compute target Q values
            # If terminal, target_q = rewards only
            # Otherwise, target_q = Eq.(6) of arXiv.1511.06581
            target_q_values = rewards + (1 - dones) * self.gamma_reward * next_q_values
            # Get current Q values, from policy network
            # Using current state and current action (not argmax)
            current_q_values = self.policy_net(states)
            current_q_values = current_q_values.gather(idx_gather, actions)
            # Compute losses with importance sampling weights
            q_loss = self.loss_func_body(current_q_values, target_q_values, reduction='none')
            # Compute priorities for replay buffer
            with torch.no_grad():
                td_errors = torch.abs(target_q_values - current_q_values).cpu().numpy()
                self.memory_buffer.update_priorities(indices, td_errors)        
            # Optimize loss function
            total_loss = torch.mean(weights * q_loss)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.grad_norm_clip,
                error_if_nonfinite=True)
            self.optimizer.step()
            self.scheduler.step()
            # Update beta for importance sampling
            self.beta = min(1.0, self.beta + self.beta_increment)
        return
    
    
    # Update target network
    def target_network_update(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return


    # Training
    def train(self, env: CSE_purified, num_episodes: int, 
              batch_size: int = 256) -> None:
        '''
        env: environment class. 

        num_episodes: number of episode
            - Each episode starts from initial state and runs until termination.

        For environment class (follows from Gynmnasium)
            - env.reset(): reset RL state to its initial value
            - env.step(action): one step of environment (apply action and update state) 
        '''
        for episode in range(num_episodes):
            # reset env at each episode
            state = env.Reset()
            total_reward = 0; done = False; step = 0
            # Alternatively one can use energy/residual criterion
            while not done and step < self.max_step:
                # select action (for a single state instead of batched) 
                action = self.select_action(state)
                # next state (after applying action), reward, done
                next_state, reward, done, E_expect = env.Step(action)
                # store transition in memory_buffer
                self.memory_buffer.push(state, action, reward, next_state, done)
                # Train if already have enough samples
                # Otherwise: only do sampling using current network
                if self.memory_buffer.ready_to_train(batch_size):
                    self.update_network(batch_size)
                # update RL state and reward
                step += 1
                total_reward += reward
                state = next_state
            ###
            # Store metrics: average rewards per episode per step is used
            self.metrics['episode_rewards'].append(total_reward/step)
            self.metrics['episode_residuals'].append(np.sum(np.abs(state)))
            self.metrics['episode_steps'].append(step)
            self.metrics['episode_energy'].append(E_expect)
            self.metrics['episode_wavefunc'].append(env.current_wf)
            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Update target network
            if episode % self.update_target_step == 0:
                self.target_network_update()
            # Print progress per n episode
            if episode % self.print_step == 0:
                if isinstance(E_expect, float):
                    print(f"Episode {episode}: Final energy: {E_expect:.8f}, Steps: {step}")
                elif isinstance(E_expect, np.ndarray):
                    E_print = '\t'.join([f"{x:.4f}" for x in E_expect.tolist()])
                    print(f"Episode {episode}: Steps: {step}, Final energy:   " + E_print)
                    #E_print = ', '.join([f"{x:.4f}" for x in E_expect.tolist()])
                    # print(f"Episode {episode}: Steps: {step}, Final energy:  [" + E_print + "]")
        # clear buffer when finished
        self.memory_buffer.clear()
        return


    # Save DQN-policy (only policy_net) for testing
    def Save_policy(self, filename: str, save_optim: bool = False) -> None:
        state_dict = {'model': self.policy_net.state_dict()}
        if save_optim:
            state_dict['optimizer'] = self.optimizer.state_dict()
            state_dict['scheduler'] = self.scheduler.state_dict()
        if not filename.endswith('.pt'): filename += '.pt'
        torch.save(state_dict, filename)
        return
        

# Minimum model for testing
class DQN_policy:
    def __init__(self, state_dim: int, num_actions: int,
                 max_step_RL: int = 5, device: torch.device = 'cpu'):
        self.policy_net = DQN_Network(state_dim, num_actions, device=device)
        self.device = device
        self.max_step = max_step_RL
        ###


    # Load policy from file or agent-obj
    def Load_policy(self, fn: str | DQN_Agent) -> None:
        if isinstance(fn, str):
            if not fn.endswith('.pt'): fn == '.pt'
            state_dict = torch.load(fn, map_location=self.device)
            self.policy_net.load_state_dict(state_dict['model'])
        elif isinstance(fn, DQN_Agent):
            self.policy_net.load_state_dict(fn.policy_net.state_dict())            
        return
 
    # select action (index of operator) based on state
    def select_action(self, state: np.ndarray) -> int:
        '''
        state: current RL state, (state_dim)

        Return:
            - action index (integer)
        '''
        # (state_dim) -> (1, state_dim)
        state = torch.as_tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
            action_index = torch.argmax(q_values, dim=-1).item()
        return action_index
    



# excited state DQN
def Train_H2_excited(H_sys_matrix: np.ndarray) -> dict:
    # H2 molecule parameters
    n_orbit = 4; K_excited = 4
    weight_vec = [0.45, 0.45, 0.05, 0.05]
    ### 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # number of orbitals
    rdm_list = CSE_action.Get_matrices_Hermitian(
        n_orbit, CSE_action.Get_indices(n_orbit))

    # initial wavefunction
    init_wf = CSE_action.Initial_ortho_wf(n_orbit, K_excited)

    # Environment setup
    env = CSE_purified(weight_vec, init_wf, H_sys_matrix, rdm_list)
    E_start = env.energy_value(env.current_wf).tolist()
    E_start = ', '.join([f"{x:.4f}" for x in E_start])
    # Energy expectation from HF state
    print ("Starting energy is: [", E_start, "]")

    # D3QN Agent
    agent = DQN_Agent(env.initial_state.shape[0], len(rdm_list), device=device)
    
    # Training
    num_episodes = 1001
    agent.train(env, num_episodes)
    # Save temporary metrics
    np.save('Metrics_H2.npy', agent.metrics)
     
    return 



if __name__ == '__main__':   
    H_sys = np.load('h2_matrix', allow_pickle=True)
    Train_H2_excited(H_sys)
    ### 


