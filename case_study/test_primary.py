#!/usr/bin/env python3
"""
Simple test for primary control without complex dependencies
"""

import numpy as np
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass 
class ControllerState:
    """State variables for primary controller"""
    frequency: float = 60.0
    active_power: float = 0.0
    reactive_power: float = 0.0
    voltage: float = 1.0
    phase_angle: float = 0.0

class SimpleDroopController:
    """Simplified droop controller for testing"""
    
    def __init__(self, node_id: int, rated_power: float = 500.0):
        self.node_id = node_id
        self.rated_power = rated_power
        self.freq_droop_gain = 0.05  # 5% droop
        self.freq_ref = 60.0
        self.power_ref = 0.0
        
        logger.info(f"Initialized droop controller for node {node_id}")
    
    def update(self, state: ControllerState, dt: float):
        """Update droop control"""
        power_error = state.active_power - self.power_ref
        freq_command = self.freq_ref - self.freq_droop_gain * power_error
        return freq_command, 1.0  # frequency, voltage

class EnhancedDroopController:
    """Enhanced controller with simple adaptive gains"""
    
    def __init__(self, node_id: int, rated_power: float = 500.0):
        self.node_id = node_id
        self.rated_power = rated_power
        self.base_droop_gain = 0.04  # Base gain (optimized)
        self.adaptive_factor = 0.1
        self.freq_ref = 60.0
        self.power_ref = 0.0
        
        logger.info(f"Initialized enhanced controller for node {node_id}")
    
    def update(self, state: ControllerState, dt: float):
        """Update enhanced control with adaptive gains"""
        
        # Adaptive gain based on frequency deviation
        freq_error = abs(state.frequency - self.freq_ref)
        adaptive_gain = self.base_droop_gain * (1 + self.adaptive_factor * freq_error)
        
        # Apply bounds
        adaptive_gain = np.clip(adaptive_gain, 0.02, 0.08)
        
        # Enhanced droop with small correction
        power_error = state.active_power - self.power_ref
        freq_command = self.freq_ref - adaptive_gain * power_error
        
        # Add simple physics-informed correction (placeholder)
        physics_correction = 0.01 * np.sin(state.phase_angle) * freq_error
        freq_command += physics_correction
        
        return freq_command, 1.0

def run_test_scenario(controller, scenario_name: str):
    """Run a single test scenario"""
    
    logger.info(f"Running scenario: {scenario_name}")
    
    # Initialize state
    state = ControllerState()
    state.frequency = 60.0
    state.active_power = 0.5  # 50% initial load
    
    # Simulation parameters
    dt = 0.01
    simulation_time = 20.0
    steps = int(simulation_time / dt)
    
    # Data logging
    times = []
    frequencies = []
    powers = []
    
    for step in range(steps):
        t = step * dt
        
        # Apply disturbance at t=10s
        if abs(t - 10.0) < dt/2:
            state.active_power += 0.2  # 20% load step
            logger.info(f"Applied load step: +20% at t={t:.2f}s")
        
        # Update controller  
        freq_cmd, volt_cmd = controller.update(state, dt)
        
        # Simple first-order system response
        tau = 0.2  # Time constant
        state.frequency += dt/tau * (freq_cmd - state.frequency)
        
        # Log every 100ms
        if step % 10 == 0:
            times.append(t)
            frequencies.append(state.frequency)
            powers.append(state.active_power)
    
    # Calculate performance metrics
    disturbance_idx = int(10.0 / dt / 10)  # Account for 100ms logging interval
    print(f"DEBUG: Total samples: {len(frequencies)}, Disturbance index: {disturbance_idx}")
    
    if disturbance_idx < len(frequencies):
        post_disturbance_freq = frequencies[disturbance_idx:]
        print(f"DEBUG: Post-disturbance samples: {len(post_disturbance_freq)}")
        
        if len(post_disturbance_freq) > 0:
            freq_nadir = min(post_disturbance_freq)
            final_freq = post_disturbance_freq[-1]
            print(f"DEBUG: Freq nadir: {freq_nadir:.4f}, Final freq: {final_freq:.4f}")
            
            # RoCoF calculation
            if len(post_disturbance_freq) > 1:
                freq_array = np.array(post_disturbance_freq)
                freq_diff = np.diff(freq_array)
                time_diff = 0.1  # 100ms between samples
                rocof = freq_diff / time_diff
                max_rocof = np.max(np.abs(rocof)) if len(rocof) > 0 else 0.0
            else:
                max_rocof = 0.0
            
            # Settling time (2% criterion)
            settling_threshold = 0.02 * abs(final_freq - 60.0) if final_freq != 60.0 else 0.02
            settling_time = len(post_disturbance_freq) * 0.1  # Default to full time
            
            for i, freq in enumerate(post_disturbance_freq):
                if abs(freq - final_freq) <= settling_threshold:
                    settling_time = i * 0.1  # Convert to time
                    break
            
            metrics = {
                'frequency_nadir': abs(freq_nadir - 60.0),
                'max_rocof': max_rocof,
                'settling_time': settling_time,
                'final_error': abs(final_freq - 60.0)
            }
            
            logger.info(f"Results for {scenario_name}:")
            logger.info(f"  Frequency nadir: {metrics['frequency_nadir']:.4f} Hz")
            logger.info(f"  Max RoCoF: {metrics['max_rocof']:.4f} Hz/s")  
            logger.info(f"  Settling time: {metrics['settling_time']:.2f} s")
            logger.info(f"  Final error: {metrics['final_error']:.4f} Hz")
            
            return metrics
        else:
            print("DEBUG: No post-disturbance samples found")
            return {}
    else:
        print("DEBUG: Disturbance index out of range")
        return {}

def main():
    """Main test function"""
    
    logger.info("Starting Primary Control Validation Test")
    logger.info("=" * 50)
    
    # Create controllers
    conventional = SimpleDroopController(0)
    enhanced = EnhancedDroopController(0)
    
    # Test scenarios
    scenarios = ['Load Step Response']
    
    results = {}
    
    for scenario in scenarios:
        logger.info(f"\nTesting scenario: {scenario}")
        logger.info("-" * 30)
        
        # Test conventional controller
        logger.info("Conventional Controller:")
        conv_results = run_test_scenario(conventional, f"{scenario}_Conventional")
        
        # Test enhanced controller
        logger.info("\nEnhanced Controller:")
        enh_results = run_test_scenario(enhanced, f"{scenario}_Enhanced")
        
        # Compare results
        if conv_results and enh_results:
            logger.info(f"\nComparison for {scenario}:")
            logger.info("-" * 20)
            
            # Calculate improvements
            nadir_improvement = (conv_results['frequency_nadir'] - enh_results['frequency_nadir']) / conv_results['frequency_nadir'] * 100
            rocof_improvement = (conv_results['max_rocof'] - enh_results['max_rocof']) / conv_results['max_rocof'] * 100 if conv_results['max_rocof'] > 0 else 0
            settling_improvement = (conv_results['settling_time'] - enh_results['settling_time']) / conv_results['settling_time'] * 100 if conv_results['settling_time'] > 0 else 0
            
            logger.info(f"  Frequency nadir improvement: {nadir_improvement:.1f}%")
            logger.info(f"  RoCoF improvement: {rocof_improvement:.1f}%")
            logger.info(f"  Settling time improvement: {settling_improvement:.1f}%")
            
            results[scenario] = {
                'conventional': conv_results,
                'enhanced': enh_results,
                'improvements': {
                    'nadir': nadir_improvement,
                    'rocof': rocof_improvement, 
                    'settling': settling_improvement
                }
            }
    
    logger.info("\nPrimary Control Validation Complete")
    logger.info("=" * 50)
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\nFinal results: {len(results)} scenarios tested")