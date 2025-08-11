#!/usr/bin/env python3
"""
Secondary Control Layer Implementation
====================================

This module implements both conventional consensus control and the proposed
MARL-enhanced consensus for secondary frequency and voltage restoration.

Mathematical formulation:
- Conventional: η̇ᵢ = αᵢ(ωᵢ - ω*) + βᵢ Σⱼ aᵢⱼ(ηⱼ - ηᵢ)
- Enhanced: η̇ᵢ = αᵢ(ωᵢ - ω*) + βᵢ Σⱼ aᵢⱼ(ηⱼ - ηᵢ) + f_MARL(sᵢ, aᵢ)
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SecondaryControlState:
    """State variables for secondary control"""
    frequency: float = 60.0
    voltage: float = 1.0
    active_power: float = 0.0
    reactive_power: float = 0.0
    
    # Secondary control states
    freq_integral_state: float = 0.0
    voltage_integral_state: float = 0.0
    
    # Communication states
    neighbor_freq_states: Dict[int, float] = None
    neighbor_voltage_states: Dict[int, float] = None
    
    def __post_init__(self):
        if self.neighbor_freq_states is None:
            self.neighbor_freq_states = {}
        if self.neighbor_voltage_states is None:
            self.neighbor_voltage_states = {}

class SimpleMARLAgent:
    """
    Simplified Multi-Agent Reinforcement Learning agent for consensus enhancement
    
    This is a simplified version that demonstrates the concept without full
    neural network training. In practice, this would be a trained DDPG/PPO agent.
    """
    
    def __init__(self, node_id: int, n_neighbors: int = 3):
        self.node_id = node_id
        self.n_neighbors = n_neighbors
        
        # Simplified Q-table for demonstration (state-action values)
        self.action_space_size = 5  # [-0.1, -0.05, 0, 0.05, 0.1] corrections
        self.state_discretization = 10  # Discretize continuous states
        
        # Initialize with physics-informed policy (small corrections)
        self.policy_weights = np.random.normal(0, 0.01, (4, self.action_space_size))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=1000)
        
        # Learning parameters
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.discount_factor = 0.9
        
        logger.info(f"Initialized MARL agent for node {node_id}")
    
    def get_state_features(self, state: SecondaryControlState) -> np.ndarray:
        """Extract state features for MARL decision making"""
        
        # Normalize state features
        freq_error = (state.frequency - 60.0) / 0.5  # Normalize by ±0.5 Hz
        voltage_error = (state.voltage - 1.0) / 0.1   # Normalize by ±0.1 pu
        power_level = state.active_power  # Already in pu
        
        # Calculate neighbor consensus error
        if state.neighbor_freq_states:
            avg_neighbor_freq = np.mean(list(state.neighbor_freq_states.values()))
            consensus_error = (state.frequency - avg_neighbor_freq) / 0.1
        else:
            consensus_error = 0.0
        
        features = np.array([freq_error, voltage_error, power_level, consensus_error])
        return np.clip(features, -2.0, 2.0)  # Bound features
    
    def select_action(self, state: SecondaryControlState, training: bool = False) -> Tuple[float, float]:
        """
        Select MARL action (frequency and voltage corrections)
        
        Returns: (freq_correction, voltage_correction)
        """
        
        features = self.get_state_features(state)
        
        # Simple policy: linear combination of features with learned weights
        if training and np.random.random() < self.exploration_rate:
            # Exploration: random actions
            freq_action_idx = np.random.randint(self.action_space_size)
            voltage_action_idx = np.random.randint(self.action_space_size)
        else:
            # Exploitation: use learned policy
            freq_scores = features @ self.policy_weights[:, :]
            voltage_scores = features @ self.policy_weights[:, :]
            
            freq_action_idx = np.argmax(np.mean(freq_scores))
            voltage_action_idx = np.argmax(np.mean(voltage_scores))
        
        # Convert action indices to actual corrections
        action_values = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
        freq_correction = action_values[min(freq_action_idx, len(action_values)-1)]
        voltage_correction = action_values[min(voltage_action_idx, len(action_values)-1)]
        
        # Apply physics-informed bounds (inertia-aware)
        freq_error = abs(state.frequency - 60.0)
        if freq_error > 0.2:  # Large deviation - more aggressive correction
            freq_correction *= 1.5
        elif freq_error < 0.05:  # Small deviation - gentle correction
            freq_correction *= 0.5
        
        return freq_correction, voltage_correction
    
    def update_policy(self, state: SecondaryControlState, action: Tuple[float, float],
                     reward: float, next_state: SecondaryControlState):
        """
        Simple policy update based on reward feedback
        (In practice, this would be a full DDPG/PPO update)
        """
        
        # Store experience
        experience = (state, action, reward, next_state)
        self.experience_buffer.append(experience)
        
        # Simple gradient ascent on policy weights
        features = self.get_state_features(state)
        
        # Reward-weighted feature update
        if reward > 0:  # Good action - reinforce
            self.policy_weights += self.learning_rate * reward * np.outer(features, np.ones(self.action_space_size))
        else:  # Bad action - discourage
            self.policy_weights -= self.learning_rate * abs(reward) * np.outer(features, np.ones(self.action_space_size))
        
        # Bound weights to prevent instability
        self.policy_weights = np.clip(self.policy_weights, -1.0, 1.0)

class ConventionalConsensusController:
    """Baseline conventional consensus controller for secondary control"""
    
    def __init__(self, node_id: int, adjacency_matrix: np.ndarray):
        self.node_id = node_id
        self.adjacency = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        
        # Consensus control gains
        self.freq_integral_gain = 5.0    # αᵢ for frequency
        self.freq_consensus_gain = 2.0   # βᵢ for frequency
        self.voltage_integral_gain = 3.0 # αᵢ for voltage
        self.voltage_consensus_gain = 1.5 # βᵢ for voltage
        
        # Reference setpoints
        self.freq_ref = 60.0
        self.voltage_ref = 1.0
        
        logger.info(f"Initialized conventional consensus controller for node {node_id}")
    
    def update(self, state: SecondaryControlState, dt: float) -> Tuple[float, float]:
        """
        Update conventional consensus control
        
        Implements: η̇ᵢ = αᵢ(ωᵢ - ω*) + βᵢ Σⱼ aᵢⱼ(ηⱼ - ηᵢ)
        
        Returns: (freq_correction, voltage_correction)
        """
        
        # Frequency consensus
        freq_error = state.frequency - self.freq_ref
        
        # Calculate neighbor consensus term
        freq_consensus_term = 0.0
        if state.neighbor_freq_states:
            for neighbor_id, neighbor_freq_state in state.neighbor_freq_states.items():
                if neighbor_id < self.n_nodes and self.adjacency[self.node_id, neighbor_id] > 0:
                    freq_consensus_term += self.adjacency[self.node_id, neighbor_id] * \
                                          (neighbor_freq_state - state.freq_integral_state)
        
        # Update integral state
        freq_integral_update = (self.freq_integral_gain * freq_error + 
                               self.freq_consensus_gain * freq_consensus_term) * dt
        
        # Voltage consensus (similar structure)
        voltage_error = state.voltage - self.voltage_ref
        
        voltage_consensus_term = 0.0
        if state.neighbor_voltage_states:
            for neighbor_id, neighbor_voltage_state in state.neighbor_voltage_states.items():
                if neighbor_id < self.n_nodes and self.adjacency[self.node_id, neighbor_id] > 0:
                    voltage_consensus_term += self.adjacency[self.node_id, neighbor_id] * \
                                            (neighbor_voltage_state - state.voltage_integral_state)
        
        voltage_integral_update = (self.voltage_integral_gain * voltage_error + 
                                  self.voltage_consensus_gain * voltage_consensus_term) * dt
        
        return freq_integral_update, voltage_integral_update

class MARLEnhancedConsensusController:
    """Enhanced consensus controller with MARL augmentation"""
    
    def __init__(self, node_id: int, adjacency_matrix: np.ndarray):
        self.node_id = node_id
        self.adjacency = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        
        # Base consensus gains (optimized for MARL cooperation)
        self.freq_integral_gain = 4.0    # Slightly reduced for MARL cooperation
        self.freq_consensus_gain = 1.8
        self.voltage_integral_gain = 2.5
        self.voltage_consensus_gain = 1.2
        
        # Initialize MARL agent
        n_neighbors = np.sum(adjacency_matrix[node_id, :])
        self.marl_agent = SimpleMARLAgent(node_id, n_neighbors)
        
        # Event-triggered communication parameters
        self.comm_threshold_freq = 0.01  # Hz
        self.comm_threshold_voltage = 0.005  # pu
        self.last_comm_freq = 60.0
        self.last_comm_voltage = 1.0
        
        # Performance tracking for reward calculation
        self.performance_history = deque(maxlen=100)
        
        self.freq_ref = 60.0
        self.voltage_ref = 1.0
        
        logger.info(f"Initialized MARL-enhanced consensus controller for node {node_id}")
    
    def should_communicate(self, state: SecondaryControlState) -> bool:
        """
        Event-triggered communication decision
        
        Reduces communication overhead while maintaining performance
        """
        
        freq_change = abs(state.frequency - self.last_comm_freq)
        voltage_change = abs(state.voltage - self.last_comm_voltage)
        
        if (freq_change > self.comm_threshold_freq or 
            voltage_change > self.comm_threshold_voltage):
            self.last_comm_freq = state.frequency
            self.last_comm_voltage = state.voltage
            return True
        
        return False
    
    def calculate_reward(self, prev_state: SecondaryControlState, 
                        current_state: SecondaryControlState) -> float:
        """
        Calculate reward for MARL learning
        
        Reward based on:
        - Frequency/voltage error reduction
        - Consensus achievement  
        - Settling time improvement
        """
        
        # Frequency error reduction reward
        prev_freq_error = abs(prev_state.frequency - self.freq_ref)
        curr_freq_error = abs(current_state.frequency - self.freq_ref)
        freq_reward = (prev_freq_error - curr_freq_error) * 10.0
        
        # Voltage error reduction reward
        prev_voltage_error = abs(prev_state.voltage - self.voltage_ref)
        curr_voltage_error = abs(current_state.voltage - self.voltage_ref)
        voltage_reward = (prev_voltage_error - curr_voltage_error) * 5.0
        
        # Consensus reward (minimize deviation from neighbors)
        consensus_reward = 0.0
        if current_state.neighbor_freq_states:
            neighbor_freqs = list(current_state.neighbor_freq_states.values())
            if neighbor_freqs:
                avg_neighbor_freq = np.mean(neighbor_freqs)
                consensus_error = abs(current_state.frequency - avg_neighbor_freq)
                consensus_reward = -consensus_error * 2.0  # Penalty for deviation
        
        # Stability reward (penalize oscillations)
        stability_reward = 0.0
        if len(self.performance_history) > 1:
            recent_freq_std = np.std([h['frequency'] for h in list(self.performance_history)[-10:]])
            stability_reward = -recent_freq_std * 1.0
        
        total_reward = freq_reward + voltage_reward + consensus_reward + stability_reward
        return np.clip(total_reward, -10.0, 10.0)  # Bound rewards
    
    def update(self, state: SecondaryControlState, dt: float, 
              training: bool = False) -> Tuple[float, float]:
        """
        Update MARL-enhanced consensus control
        
        Implements: η̇ᵢ = αᵢ(ωᵢ - ω*) + βᵢ Σⱼ aᵢⱼ(ηⱼ - ηᵢ) + f_MARL(sᵢ, aᵢ)
        """
        
        # Step 1: Conventional consensus control
        freq_error = state.frequency - self.freq_ref
        voltage_error = state.voltage - self.voltage_ref
        
        # Frequency consensus
        freq_consensus_term = 0.0
        if state.neighbor_freq_states:
            for neighbor_id, neighbor_freq_state in state.neighbor_freq_states.items():
                if neighbor_id < self.n_nodes and self.adjacency[self.node_id, neighbor_id] > 0:
                    freq_consensus_term += self.adjacency[self.node_id, neighbor_id] * \
                                          (neighbor_freq_state - state.freq_integral_state)
        
        freq_base_update = (self.freq_integral_gain * freq_error + 
                           self.freq_consensus_gain * freq_consensus_term) * dt
        
        # Voltage consensus
        voltage_consensus_term = 0.0
        if state.neighbor_voltage_states:
            for neighbor_id, neighbor_voltage_state in state.neighbor_voltage_states.items():
                if neighbor_id < self.n_nodes and self.adjacency[self.node_id, neighbor_id] > 0:
                    voltage_consensus_term += self.adjacency[self.node_id, neighbor_id] * \
                                            (neighbor_voltage_state - state.voltage_integral_state)
        
        voltage_base_update = (self.voltage_integral_gain * voltage_error + 
                              self.voltage_consensus_gain * voltage_consensus_term) * dt
        
        # Step 2: MARL enhancement
        marl_freq_correction, marl_voltage_correction = self.marl_agent.select_action(state, training)
        
        # Step 3: Combine base control with MARL corrections
        freq_update = freq_base_update + marl_freq_correction * dt
        voltage_update = voltage_base_update + marl_voltage_correction * dt
        
        # Step 4: Update performance tracking and MARL learning
        current_performance = {
            'frequency': state.frequency,
            'voltage': state.voltage,
            'freq_error': abs(freq_error),
            'voltage_error': abs(voltage_error)
        }
        
        if training and len(self.performance_history) > 0:
            # Calculate reward and update policy
            prev_performance = self.performance_history[-1]
            prev_state_approx = SecondaryControlState(
                frequency=prev_performance['frequency'],
                voltage=prev_performance['voltage']
            )
            
            reward = self.calculate_reward(prev_state_approx, state)
            action = (marl_freq_correction, marl_voltage_correction)
            self.marl_agent.update_policy(prev_state_approx, action, reward, state)
        
        self.performance_history.append(current_performance)
        
        return freq_update, voltage_update

def run_secondary_control_test():
    """Test secondary control implementations"""
    
    logger.info("Testing Secondary Control Layer")
    logger.info("=" * 50)
    
    # Create 4-node network adjacency matrix
    adjacency = np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1], 
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])
    
    # Test scenario: voltage/frequency restoration after disturbance
    simulation_time = 15.0  # seconds
    dt = 0.01
    steps = int(simulation_time / dt)
    
    results = {}
    
    for controller_type in ['conventional', 'marl_enhanced']:
        logger.info(f"\nTesting {controller_type} controller:")
        logger.info("-" * 30)
        
        # Initialize controllers for all nodes
        controllers = []
        states = []
        
        for node_id in range(4):
            if controller_type == 'conventional':
                controller = ConventionalConsensusController(node_id, adjacency)
            else:
                controller = MARLEnhancedConsensusController(node_id, adjacency)
            
            controllers.append(controller)
            states.append(SecondaryControlState())
        
        # Initialize with slight frequency deviation (simulating post-primary-control state)
        for i, state in enumerate(states):
            state.frequency = 60.0 - 0.05 * (i + 1)  # Different initial deviations
            state.voltage = 1.0 - 0.01 * i
        
        # Data logging
        time_history = []
        freq_histories = [[] for _ in range(4)]
        voltage_histories = [[] for _ in range(4)]
        
        # Simulation loop
        for step in range(steps):
            t = step * dt
            
            # Update neighbor information (with communication delay simulation)
            for i, state in enumerate(states):
                state.neighbor_freq_states = {}
                state.neighbor_voltage_states = {}
                
                for j in range(4):
                    if adjacency[i, j] > 0:  # Connected neighbors
                        # Simple delay model: use previous timestep data
                        delay_steps = max(1, int(0.05 / dt))  # 50ms delay
                        if step >= delay_steps:
                            delayed_step = step - delay_steps
                            if delayed_step < len(freq_histories[j]):
                                state.neighbor_freq_states[j] = freq_histories[j][delayed_step] if freq_histories[j] else states[j].frequency
                                state.neighbor_voltage_states[j] = voltage_histories[j][delayed_step] if voltage_histories[j] else states[j].voltage
            
            # Apply disturbance at t=5s (additional frequency drop)
            if abs(t - 5.0) < dt/2:
                for state in states:
                    state.frequency -= 0.1  # Additional 0.1 Hz drop
                logger.info(f"Applied frequency disturbance at t={t:.2f}s")
            
            # Update controllers
            for i, (controller, state) in enumerate(zip(controllers, states)):
                if controller_type == 'marl_enhanced':
                    freq_update, voltage_update = controller.update(state, dt, training=True)
                else:
                    freq_update, voltage_update = controller.update(state, dt)
                
                # Apply updates to integral states (these feed back to primary control)
                state.freq_integral_state += freq_update
                state.voltage_integral_state += voltage_update
                
                # Simple first-order restoration dynamics
                tau_freq = 1.0  # 1 second time constant for frequency restoration
                tau_voltage = 0.5  # 0.5 second time constant for voltage restoration
                
                freq_target = 60.0 - 0.1 * state.freq_integral_state  # Secondary control influence
                voltage_target = 1.0 - 0.05 * state.voltage_integral_state
                
                state.frequency += dt/tau_freq * (freq_target - state.frequency)
                state.voltage += dt/tau_voltage * (voltage_target - state.voltage)
            
            # Log data every 10ms
            if step % 1 == 0:
                time_history.append(t)
                for i, state in enumerate(states):
                    freq_histories[i].append(state.frequency)
                    voltage_histories[i].append(state.voltage)
        
        # Calculate performance metrics
        disturbance_idx = int(5.0 / dt)  # Disturbance at 5 seconds
        
        if disturbance_idx < len(time_history):
            # Calculate average system frequency deviation
            avg_freq_history = []
            for step_idx in range(len(time_history)):
                avg_freq = np.mean([freq_histories[i][step_idx] for i in range(4) if step_idx < len(freq_histories[i])])
                avg_freq_history.append(avg_freq)
            
            post_disturbance_freq = avg_freq_history[disturbance_idx:]
            
            if len(post_disturbance_freq) > 0:
                # Settling time calculation (2% of 60 Hz = 0.02 Hz)
                settling_threshold = 0.02
                final_freq = post_disturbance_freq[-1]
                
                settling_time = len(post_disturbance_freq) * dt  # Default to full time
                for i, freq in enumerate(post_disturbance_freq):
                    if abs(freq - 60.0) <= settling_threshold:
                        settling_time = i * dt
                        break
                
                # Maximum frequency deviation
                max_deviation = max([abs(f - 60.0) for f in post_disturbance_freq])
                
                # Average restoration rate
                if len(post_disturbance_freq) > 1:
                    initial_deviation = abs(post_disturbance_freq[0] - 60.0)
                    final_deviation = abs(post_disturbance_freq[-1] - 60.0)
                    restoration_rate = (initial_deviation - final_deviation) / (len(post_disturbance_freq) * dt)
                else:
                    restoration_rate = 0.0
                
                results[controller_type] = {
                    'settling_time': settling_time,
                    'max_deviation': max_deviation,
                    'final_error': abs(final_freq - 60.0),
                    'restoration_rate': restoration_rate
                }
                
                logger.info(f"Results:")
                logger.info(f"  Settling time: {settling_time:.2f} s")
                logger.info(f"  Max deviation: {max_deviation:.4f} Hz") 
                logger.info(f"  Final error: {abs(final_freq - 60.0):.4f} Hz")
                logger.info(f"  Restoration rate: {restoration_rate:.4f} Hz/s")
    
    # Compare results
    if 'conventional' in results and 'marl_enhanced' in results:
        conv = results['conventional']
        marl = results['marl_enhanced']
        
        settling_improvement = (conv['settling_time'] - marl['settling_time']) / conv['settling_time'] * 100
        deviation_improvement = (conv['max_deviation'] - marl['max_deviation']) / conv['max_deviation'] * 100
        restoration_improvement = (marl['restoration_rate'] - conv['restoration_rate']) / conv['restoration_rate'] * 100 if conv['restoration_rate'] > 0 else 0
        
        logger.info(f"\nSecondary Control Performance Comparison:")
        logger.info("=" * 50)
        logger.info(f"  Settling time improvement: {settling_improvement:.1f}%")
        logger.info(f"  Max deviation improvement: {deviation_improvement:.1f}%")  
        logger.info(f"  Restoration rate improvement: {restoration_improvement:.1f}%")
    
    return results

def main():
    """Main test function"""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    return run_secondary_control_test()

if __name__ == "__main__":
    results = main()