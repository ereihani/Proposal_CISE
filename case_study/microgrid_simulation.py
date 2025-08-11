#!/usr/bin/env python3
"""
Campus Microgrid Control Validation Case Study
==============================================

This simulation framework implements and validates the proposed BITW controller
approach for low-inertia campus microgrids, comparing performance against
conventional control methods.

Authors: [Research Team]
Date: August 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import json
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MicrogridNode:
    """Represents a single microgrid node (inverter-based resource)"""
    node_id: int
    node_type: str  # 'pv_battery', 'wind_battery', 'grid_tie'
    rated_power: float  # kW
    location: Tuple[float, float]  # (x, y) coordinates
    
    # Electrical parameters
    voltage_nominal: float = 480.0  # V (line-to-line)
    frequency_nominal: float = 60.0  # Hz
    
    # Control parameters
    droop_gain: float = 0.05  # Hz/pu
    voltage_droop_gain: float = 0.02  # V/pu
    inertia_constant: float = 5.0  # s
    damping_coefficient: float = 1.0  # pu/s
    
    # State variables (initialized)
    frequency: float = 60.0
    voltage: float = 1.0  # pu
    active_power: float = 0.0  # pu
    reactive_power: float = 0.0  # pu
    phase_angle: float = 0.0  # rad

@dataclass
class NetworkLine:
    """Represents transmission line between nodes"""
    from_node: int
    to_node: int
    resistance: float  # pu
    reactance: float  # pu
    susceptance: float = 0.0  # pu

class CampusMicrogridSystem:
    """
    4-node campus microgrid system for validation studies
    
    Topology:
    - Node 0: Solar PV + Battery (500kW)  
    - Node 1: Solar PV + Battery (500kW)
    - Node 2: Wind + Battery (300kW)
    - Node 3: Grid connection point (1000kW)
    """
    
    def __init__(self):
        self.time = 0.0
        self.dt = 0.01  # 10ms time step
        self.setup_network_topology()
        self.setup_load_profiles()
        self.results_history = []
        
        # Communication delay parameters
        self.comm_delay = 0.05  # 50ms baseline delay
        self.delay_buffer = {}  # Store delayed messages
        
        logger.info("Initialized 4-node campus microgrid system")
    
    def setup_network_topology(self):
        """Initialize the 4-node microgrid topology"""
        
        # Create nodes
        self.nodes = [
            MicrogridNode(0, 'pv_battery', 500.0, (0, 0)),
            MicrogridNode(1, 'pv_battery', 500.0, (1, 0)),  
            MicrogridNode(2, 'wind_battery', 300.0, (0, 1)),
            MicrogridNode(3, 'grid_tie', 1000.0, (1, 1))
        ]
        
        # Create network lines (simple mesh topology)
        self.lines = [
            NetworkLine(0, 1, 0.02, 0.08),  # Node 0-1
            NetworkLine(0, 2, 0.03, 0.12),  # Node 0-2  
            NetworkLine(1, 3, 0.02, 0.08),  # Node 1-3
            NetworkLine(2, 3, 0.03, 0.12),  # Node 2-3
        ]
        
        # Create adjacency matrix for communication network
        self.comm_adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1], 
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])
        
        logger.info(f"Network topology: {len(self.nodes)} nodes, {len(self.lines)} lines")
    
    def setup_load_profiles(self):
        """Setup realistic campus load profiles (CSUB student center based)"""
        
        # Typical hourly load profile for campus building (normalized to 1.0 peak)
        hourly_profile = np.array([
            0.4, 0.35, 0.3, 0.3, 0.35, 0.5,   # 00:00-05:00 (night/early morning)
            0.7, 0.85, 0.95, 1.0, 0.95, 0.9,  # 06:00-11:00 (morning ramp)  
            0.85, 0.8, 0.85, 0.9, 0.95, 0.9,  # 12:00-17:00 (afternoon)
            0.8, 0.7, 0.6, 0.55, 0.5, 0.45    # 18:00-23:00 (evening)
        ])
        
        # Base load demands for each node (kW)
        self.base_loads = {
            0: 200,  # Research building  
            1: 300,  # Student center
            2: 150,  # Dormitory
            3: 100   # Administrative
        }
        
        # Generate 24-hour load profile with 1-minute resolution
        time_hours = np.linspace(0, 24, 1441)  # 1440 minutes + 1
        self.load_profiles = {}
        
        for node_id, base_load in self.base_loads.items():
            # Interpolate hourly to minute resolution
            load_profile = np.interp(time_hours, np.arange(24), hourly_profile)
            # Add realistic noise (±5%)
            noise = np.random.normal(0, 0.05, len(load_profile))
            load_profile = load_profile * (1 + noise)
            # Scale by base load
            self.load_profiles[node_id] = base_load * load_profile
        
        logger.info("Load profiles generated for 24-hour simulation")
    
    def get_current_loads(self) -> Dict[int, float]:
        """Get current load demands at simulation time"""
        
        # Convert simulation time to hours (handle wrap-around)
        time_hours = (self.time / 3600) % 24
        time_index = int((time_hours / 24) * 1440)  # Convert to minute index
        
        current_loads = {}
        for node_id in self.load_profiles.keys():
            if time_index < len(self.load_profiles[node_id]):
                current_loads[node_id] = self.load_profiles[node_id][time_index]
            else:
                current_loads[node_id] = self.base_loads[node_id]  # Fallback
        
        return current_loads
    
    def network_power_flow(self) -> np.ndarray:
        """
        Calculate power flow through network using linearized DC approximation
        Returns: Power injection vector [P0, P1, P2, P3]
        """
        
        # Build admittance matrix
        n_nodes = len(self.nodes)
        Y = np.zeros((n_nodes, n_nodes), dtype=complex)
        
        for line in self.lines:
            i, j = line.from_node, line.to_node
            z = line.resistance + 1j * line.reactance
            y = 1.0 / z
            
            Y[i, i] += y
            Y[j, j] += y  
            Y[i, j] -= y
            Y[j, i] -= y
        
        # Extract phase angles and calculate power flows
        angles = np.array([node.phase_angle for node in self.nodes])
        
        # Simplified DC power flow: P = B * θ (where B is imaginary part of Y)
        B = Y.imag
        P_flow = B @ angles
        
        return P_flow
    
    def log_system_state(self):
        """Log current system state for analysis"""
        
        current_loads = self.get_current_loads()
        
        state = {
            'time': self.time,
            'nodes': [],
            'total_generation': 0.0,
            'total_load': sum(current_loads.values()),
            'frequency_deviation': 0.0
        }
        
        for i, node in enumerate(self.nodes):
            node_state = {
                'id': i,
                'frequency': node.frequency,
                'voltage': node.voltage, 
                'active_power': node.active_power,
                'reactive_power': node.reactive_power,
                'load_demand': current_loads.get(i, 0.0)
            }
            state['nodes'].append(node_state)
            state['total_generation'] += max(0, node.active_power)
        
        # Calculate average frequency deviation
        avg_freq = np.mean([node.frequency for node in self.nodes])
        state['frequency_deviation'] = abs(avg_freq - 60.0)
        
        self.results_history.append(state)
    
    def apply_disturbance(self, disturbance_type: str, magnitude: float = 0.2):
        """Apply test disturbance to the system"""
        
        if disturbance_type == 'load_step':
            # Sudden load increase at node 1 (student center)
            additional_load = magnitude * self.base_loads[1]
            self.base_loads[1] += additional_load
            logger.info(f"Applied load step: +{additional_load:.1f}kW at node 1")
        
        elif disturbance_type == 'generation_loss':
            # Trip largest generator (node 0)
            self.nodes[0].active_power = 0.0
            logger.info("Applied generation loss: Node 0 tripped")
        
        elif disturbance_type == 'communication_delay':
            # Increase communication delays
            self.comm_delay = magnitude
            logger.info(f"Applied communication delay: {magnitude*1000:.0f}ms")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics from simulation results"""
        
        if len(self.results_history) < 2:
            return {}
        
        # Extract time series data
        times = [state['time'] for state in self.results_history]
        freq_devs = [state['frequency_deviation'] for state in self.results_history]
        
        # Find disturbance application time (assume at t=10s)
        disturbance_idx = next((i for i, t in enumerate(times) if t >= 10.0), 0)
        
        if disturbance_idx == 0:
            return {}
        
        # Calculate metrics
        metrics = {}
        
        # Frequency nadir (maximum deviation after disturbance)
        post_dist_freqs = freq_devs[disturbance_idx:]
        if post_dist_freqs:
            metrics['frequency_nadir'] = max(post_dist_freqs)
        
            # Rate of change of frequency (RoCoF) - max derivative
            if len(post_dist_freqs) > 1:
                dt = times[1] - times[0]
                rocof = np.diff(post_dist_freqs) / dt
                metrics['max_rocof'] = max(abs(rocof)) if len(rocof) > 0 else 0.0
        
            # Settling time (time to reach ±2% of final value)
            final_freq = freq_devs[-1]
            settling_threshold = 0.02 * final_freq if final_freq > 0 else 0.02
            
            settling_idx = next((i for i, f in enumerate(post_dist_freqs) 
                               if abs(f - final_freq) <= settling_threshold), 
                              len(post_dist_freqs)-1)
            metrics['settling_time'] = times[disturbance_idx + settling_idx] - times[disturbance_idx]
        
        return metrics

def main():
    """Main simulation runner for preliminary validation"""
    
    logger.info("Starting Campus Microgrid Control Validation Case Study")
    
    # Initialize system
    microgrid = CampusMicrogridSystem()
    
    # Run baseline simulation
    logger.info("Running baseline scenario...")
    
    simulation_time = 30.0  # 30 seconds
    steps = int(simulation_time / microgrid.dt)
    
    for step in range(steps):
        microgrid.time = step * microgrid.dt
        
        # Log system state every 100ms
        if step % 10 == 0:
            microgrid.log_system_state()
        
        # Apply disturbance at t=10s
        if abs(microgrid.time - 10.0) < microgrid.dt/2:
            microgrid.apply_disturbance('load_step', 0.2)
        
        # Basic time step (placeholder for controller updates)
        # TODO: Implement actual controller dynamics
        
    logger.info("Simulation completed")
    
    # Calculate and display results
    metrics = microgrid.get_performance_metrics()
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    results_file = "baseline_simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'system_config': {
                'nodes': len(microgrid.nodes),
                'simulation_time': simulation_time,
                'time_step': microgrid.dt
            },
            'performance_metrics': metrics,
            'full_history': microgrid.results_history
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return microgrid, metrics

if __name__ == "__main__":
    microgrid_system, results = main()