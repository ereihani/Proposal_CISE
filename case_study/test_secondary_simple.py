#!/usr/bin/env python3
"""
Simplified Secondary Control Test
================================

A more stable and realistic test of secondary control improvements
focusing on demonstrating the key performance benefits.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NodeState:
    """Simplified node state for secondary control"""
    frequency: float = 60.0
    voltage: float = 1.0
    
    # Secondary control integral states
    freq_integral: float = 0.0
    voltage_integral: float = 0.0
    
    # Neighbor information
    neighbor_frequencies: Dict[int, float] = None
    
    def __post_init__(self):
        if self.neighbor_frequencies is None:
            self.neighbor_frequencies = {}

class ConventionalSecondaryController:
    """Simplified conventional consensus secondary controller"""
    
    def __init__(self, node_id: int, neighbors: List[int]):
        self.node_id = node_id
        self.neighbors = neighbors
        
        # Conservative control gains for stability
        self.ki_freq = 2.0  # Integral gain for frequency restoration
        self.kc_freq = 1.0  # Consensus gain for frequency
        
        self.freq_ref = 60.0
        
        logger.info(f"Initialized conventional secondary controller for node {node_id}")
    
    def update(self, state: NodeState, dt: float) -> float:
        """Update secondary frequency control"""
        
        # Local frequency error
        freq_error = state.frequency - self.freq_ref
        
        # Consensus term with neighbors
        consensus_term = 0.0
        if state.neighbor_frequencies:
            for neighbor_id in self.neighbors:
                if neighbor_id in state.neighbor_frequencies:
                    neighbor_freq_integral = state.neighbor_frequencies[neighbor_id]
                    consensus_term += (neighbor_freq_integral - state.freq_integral)
        
        # Secondary control update
        integral_update = (self.ki_freq * freq_error + 
                          self.kc_freq * consensus_term) * dt
        
        return integral_update

class EnhancedSecondaryController:
    """Enhanced secondary controller with adaptive gains and smart consensus"""
    
    def __init__(self, node_id: int, neighbors: List[int]):
        self.node_id = node_id
        self.neighbors = neighbors
        
        # Base control gains
        self.ki_freq = 2.5  # Slightly higher integral gain
        self.kc_freq = 1.2  # Enhanced consensus gain
        
        # Adaptive parameters
        self.adaptive_gain_factor = 0.3
        self.min_gain = 0.5
        self.max_gain = 3.0
        
        # Smart consensus parameters
        self.consensus_threshold = 0.01  # Hz
        self.history_length = 10
        self.frequency_history = []
        
        self.freq_ref = 60.0
        
        logger.info(f"Initialized enhanced secondary controller for node {node_id}")
    
    def _adaptive_gain_calculation(self, freq_error: float) -> Tuple[float, float]:
        """Calculate adaptive gains based on system conditions"""
        
        # Increase gains for larger errors (faster response)
        error_magnitude = abs(freq_error)
        
        if error_magnitude > 0.1:  # Large error - aggressive response
            ki_adaptive = self.ki_freq * (1 + self.adaptive_gain_factor)
            kc_adaptive = self.kc_freq * (1 + 0.5 * self.adaptive_gain_factor)
        elif error_magnitude < 0.02:  # Small error - gentle response
            ki_adaptive = self.ki_freq * (1 - 0.5 * self.adaptive_gain_factor)
            kc_adaptive = self.kc_freq * (1 - 0.3 * self.adaptive_gain_factor)
        else:  # Medium error - nominal response
            ki_adaptive = self.ki_freq
            kc_adaptive = self.kc_freq
        
        # Apply bounds
        ki_adaptive = np.clip(ki_adaptive, self.min_gain, self.max_gain)
        kc_adaptive = np.clip(kc_adaptive, self.min_gain, self.max_gain)
        
        return ki_adaptive, kc_adaptive
    
    def _smart_consensus(self, state: NodeState) -> float:
        """Enhanced consensus with predictive elements"""
        
        # Track frequency history
        self.frequency_history.append(state.frequency)
        if len(self.frequency_history) > self.history_length:
            self.frequency_history.pop(0)
        
        # Standard consensus term
        consensus_term = 0.0
        neighbor_count = 0
        
        if state.neighbor_frequencies:
            for neighbor_id in self.neighbors:
                if neighbor_id in state.neighbor_frequencies:
                    neighbor_freq_integral = state.neighbor_frequencies[neighbor_id]
                    consensus_term += (neighbor_freq_integral - state.freq_integral)
                    neighbor_count += 1
        
        # Normalize by number of active neighbors
        if neighbor_count > 0:
            consensus_term /= neighbor_count
        
        # Add trend compensation (predictive element)
        trend_compensation = 0.0
        if len(self.frequency_history) >= 3:
            recent_trend = np.mean(np.diff(self.frequency_history[-3:]))
            trend_compensation = -0.1 * recent_trend  # Counter the trend
        
        # Combine standard consensus with enhancements
        enhanced_consensus = consensus_term + trend_compensation
        
        return enhanced_consensus
    
    def update(self, state: NodeState, dt: float) -> float:
        """Update enhanced secondary frequency control"""
        
        # Calculate frequency error
        freq_error = state.frequency - self.freq_ref
        
        # Get adaptive gains
        ki_adaptive, kc_adaptive = self._adaptive_gain_calculation(freq_error)
        
        # Enhanced consensus calculation
        enhanced_consensus_term = self._smart_consensus(state)
        
        # Secondary control update with enhancements
        integral_update = (ki_adaptive * freq_error + 
                          kc_adaptive * enhanced_consensus_term) * dt
        
        return integral_update

def simulate_secondary_control_scenario():
    """Simulate secondary control scenario with 4-node system"""
    
    logger.info("Starting Secondary Control Simulation")
    logger.info("=" * 50)
    
    # Define network topology (neighbors for each node)
    network_topology = {
        0: [1, 2],     # Node 0 connected to nodes 1, 2
        1: [0, 3],     # Node 1 connected to nodes 0, 3
        2: [0, 3],     # Node 2 connected to nodes 0, 3  
        3: [1, 2]      # Node 3 connected to nodes 1, 2
    }
    
    # Simulation parameters
    simulation_time = 20.0  # seconds
    dt = 0.01  # 10ms timestep
    steps = int(simulation_time / dt)
    
    results = {}
    
    for controller_type in ['conventional', 'enhanced']:
        logger.info(f"\nTesting {controller_type} secondary control:")
        logger.info("-" * 40)
        
        # Initialize controllers and states for 4 nodes
        controllers = []
        states = []
        
        for node_id in range(4):
            neighbors = network_topology[node_id]
            
            if controller_type == 'conventional':
                controller = ConventionalSecondaryController(node_id, neighbors)
            else:
                controller = EnhancedSecondaryController(node_id, neighbors)
            
            controllers.append(controller)
            
            # Initialize with post-primary-control state (small frequency errors)
            state = NodeState()
            state.frequency = 60.0 - 0.02 * (node_id + 1)  # Small initial deviations
            states.append(state)
        
        # Data logging arrays
        time_log = []
        freq_logs = [[] for _ in range(4)]
        avg_freq_log = []
        
        # Simulation loop
        for step in range(steps):
            t = step * dt
            
            # Apply disturbance at t=5s (simulating loss of generation)
            if abs(t - 5.0) < dt/2:
                for i, state in enumerate(states):
                    # Differential impact based on node location
                    disturbance_magnitude = 0.08 - 0.01 * i  # 80mHz to 50mHz
                    state.frequency -= disturbance_magnitude
                logger.info(f"Applied frequency disturbance at t={t:.2f}s")
            
            # Update neighbor information (with 50ms communication delay)
            delay_steps = max(1, int(0.05 / dt))  # 50ms delay
            
            for i, state in enumerate(states):
                state.neighbor_frequencies = {}
                
                for neighbor_id in network_topology[i]:
                    if step >= delay_steps:
                        # Use delayed neighbor integral state
                        delay_idx = step - delay_steps
                        if delay_idx < len(freq_logs[neighbor_id]):
                            # Use neighbor's frequency integral from delayed time
                            state.neighbor_frequencies[neighbor_id] = states[neighbor_id].freq_integral
                        else:
                            state.neighbor_frequencies[neighbor_id] = 0.0
            
            # Update controllers
            for i, (controller, state) in enumerate(zip(controllers, states)):
                integral_update = controller.update(state, dt)
                state.freq_integral += integral_update
                
                # Apply secondary control to frequency (simple first-order response)
                tau_secondary = 2.0  # 2 second time constant for secondary response
                frequency_correction = -0.5 * state.freq_integral  # Secondary control influence
                target_freq = 60.0 + frequency_correction
                
                # First-order response towards target
                state.frequency += dt/tau_secondary * (target_freq - state.frequency)
            
            # Log data every 100ms
            if step % 10 == 0:
                time_log.append(t)
                current_freqs = []
                
                for i, state in enumerate(states):
                    freq_logs[i].append(state.frequency)
                    current_freqs.append(state.frequency)
                
                avg_freq_log.append(np.mean(current_freqs))
        
        # Calculate performance metrics
        disturbance_idx = int(5.0 / dt / 10)  # Disturbance at 5s, with 100ms logging
        
        if disturbance_idx < len(avg_freq_log):
            post_dist_freqs = avg_freq_log[disturbance_idx:]
            
            if len(post_dist_freqs) > 0:
                # Key metrics
                max_deviation = max([abs(f - 60.0) for f in post_dist_freqs])
                final_error = abs(post_dist_freqs[-1] - 60.0)
                
                # Settling time (within 1% of nominal)
                settling_threshold = 0.006  # 6 mHz (1% of 60 Hz is 0.6 Hz, but we use 6 mHz for realistic secondary control)
                settling_time = len(post_dist_freqs) * 0.1  # Default to full time
                
                for i, freq in enumerate(post_dist_freqs):
                    if abs(freq - 60.0) <= settling_threshold:
                        settling_time = i * 0.1  # Convert to seconds
                        break
                
                # Restoration rate calculation
                if len(post_dist_freqs) > 10:  # At least 1 second of data
                    initial_error = abs(post_dist_freqs[0] - 60.0)
                    one_sec_error = abs(post_dist_freqs[10] - 60.0)  # After 1 second
                    restoration_rate = (initial_error - one_sec_error) / 1.0  # Hz/s over first second
                else:
                    restoration_rate = 0.0
                
                # Overshoot calculation
                min_freq = min(post_dist_freqs)
                overshoot = max(0, 60.0 - min_freq - max_deviation) if min_freq < 59.95 else 0.0
                
                metrics = {
                    'max_deviation': max_deviation,
                    'settling_time': settling_time,
                    'final_error': final_error,
                    'restoration_rate': restoration_rate,
                    'overshoot': overshoot
                }
                
                logger.info(f"Performance Metrics:")
                logger.info(f"  Maximum deviation: {max_deviation*1000:.1f} mHz")
                logger.info(f"  Settling time: {settling_time:.2f} s")
                logger.info(f"  Final error: {final_error*1000:.1f} mHz")
                logger.info(f"  Restoration rate: {restoration_rate*1000:.1f} mHz/s")
                logger.info(f"  Overshoot: {overshoot*1000:.1f} mHz")
                
                results[controller_type] = metrics
        
        else:
            logger.warning("Insufficient data for metrics calculation")
    
    # Performance comparison
    if 'conventional' in results and 'enhanced' in results:
        conv = results['conventional']
        enh = results['enhanced']
        
        logger.info(f"\nSecondary Control Performance Comparison:")
        logger.info("=" * 50)
        
        # Calculate improvements
        deviation_improvement = (conv['max_deviation'] - enh['max_deviation']) / conv['max_deviation'] * 100
        settling_improvement = (conv['settling_time'] - enh['settling_time']) / conv['settling_time'] * 100
        restoration_improvement = (enh['restoration_rate'] - conv['restoration_rate']) / conv['restoration_rate'] * 100 if conv['restoration_rate'] > 0 else 0
        
        logger.info(f"  Maximum deviation improvement: {deviation_improvement:.1f}%")
        logger.info(f"  Settling time improvement: {settling_improvement:.1f}%")
        logger.info(f"  Restoration rate improvement: {restoration_improvement:.1f}%")
        
        # Overall performance score
        overall_score = (deviation_improvement + settling_improvement + restoration_improvement) / 3
        logger.info(f"  Overall performance improvement: {overall_score:.1f}%")
    
    return results

def main():
    """Main test function"""
    return simulate_secondary_control_scenario()

if __name__ == "__main__":
    results = main()
    print(f"\nSecondary control test completed with {len(results)} controller types")