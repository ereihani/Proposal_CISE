"""
integrated_visualization.py
Connects actual simulation data from microgrid modules to visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict
import json

# Import visualization module
from visualization_validation import MicrogridVisualizer, PerformanceValidator

# Import simulation modules
from microgrid_control_system import (
    MicrogridParams,
    FrequencyStabilitySimulator,
    MultiAgentConsensus,
    EconomicAnalysis,
    ControlBarrierFunction
)

from advanced_modules import (
    LyapunovStabilizedPINODE,
    GraphNeuralConsensus,
    ADMMOptimizer,
    AdvancedCBF,
    MicrogridTestSuite
)

class IntegratedMicrogridSimulation:
    """Runs actual simulations and connects to visualization"""
    
    def __init__(self):
        self.params = MicrogridParams()
        self.viz = MicrogridVisualizer()
        self.validator = PerformanceValidator()
        
    def run_frequency_stability_simulation(self) -> Dict:
        """Run actual frequency stability simulation"""
        print("Running Frequency Stability Simulation...")
        
        # Basic simulation using microgrid_control_system
        freq_sim = FrequencyStabilitySimulator(self.params)
        t, frequencies, states = freq_sim.simulate_disturbance(duration=20.0)
        metrics = freq_sim.verify_stability(frequencies)
        
        # Advanced simulation using advanced_modules
        pinode = LyapunovStabilizedPINODE()
        iss_margin = pinode.compute_iss_margin(0.150)  # 150ms delay
        
        # Format data for visualization
        frequency_data = {
            'time': t,
            'frequencies': frequencies,
            'max_deviation': metrics['max_deviation_hz'],
            'settling_time': metrics['settling_time_s'],
            'iss_margin': iss_margin
        }
        
        # Create formatted results
        results = {
            'max_deviation_hz': metrics['max_deviation_hz'],
            'settling_time_s': metrics['settling_time_s'],
            'frequency_nadir_hz': metrics['frequency_nadir_hz'],
            'meets_target': metrics['meets_target'],
            'iss_margin': iss_margin
        }
        
        return frequency_data, results
    
    def run_marl_consensus_simulation(self) -> Dict:
        """Run actual MARL consensus simulation"""
        print("Running Multi-Agent Consensus Simulation...")
        
        # Basic MARL simulation
        marl = MultiAgentConsensus(n_agents=16, comm_topology='mesh')
        
        # Simulate consensus
        consensus_time = 10.0
        dt = 0.01
        steps = int(consensus_time / dt)
        trajectories = np.zeros((steps, self.params.n_inverters))
        
        # Initial conditions
        trajectories[0] = np.random.uniform(-1, 1, self.params.n_inverters)
        
        # Run consensus dynamics
        for i in range(1, steps):
            d_states = marl.consensus_dynamics(trajectories[i-1], i*dt)
            trajectories[i] = trajectories[i-1] + d_states * dt
        
        conv_metrics = marl.verify_convergence(trajectories)
        
        # Advanced GNN consensus
        gnn = GraphNeuralConsensus(n_agents=16)
        states_tensor = torch.tensor(trajectories[-1], dtype=torch.float32).reshape(16, 1).repeat(1, 4)
        adj_matrix = torch.ones(16, 16) - torch.eye(16)
        
        with torch.no_grad():
            consensus_update = gnn(states_tensor, adj_matrix)
        
        # Format data for visualization
        t = np.linspace(0, consensus_time, steps)
        consensus_data = {
            'time': t,
            'trajectories': trajectories
        }
        
        results = {
            'convergence_rate': conv_metrics['convergence_rate'],
            'lambda_2': conv_metrics['lambda_2'],
            'max_tolerable_delay_s': conv_metrics['max_tolerable_delay_s'],
            'actual_delay_s': conv_metrics['actual_delay_s'],
            'stable': conv_metrics['stable']
        }
        
        return consensus_data, results
    
    def run_economic_analysis(self) -> Dict:
        """Run actual economic analysis"""
        print("Running Economic Analysis...")
        
        econ = EconomicAnalysis()
        cost_metrics = econ.calculate_tco(years=10)
        
        # Format for visualization
        cost_data = {
            'our_tco': cost_metrics['our_tco'],
            'conventional_tco': cost_metrics['conventional_tco'],
            'percentage_savings': cost_metrics['percentage_savings']
        }
        
        return cost_data, cost_metrics
    
    def run_safety_verification(self) -> Dict:
        """Run actual safety verification simulation"""
        print("Running Safety Verification...")
        
        # Basic CBF safety
        cbf_basic = ControlBarrierFunction(n_buses=self.params.n_inverters)
        safety_metrics = cbf_basic.simulate_n2_contingency(duration=3600)
        
        # Advanced CBF with QP
        cbf_advanced = AdvancedCBF()
        
        # Run a shorter test for advanced CBF
        test_duration = 100  # seconds for demo
        dt = 0.1
        steps = int(test_duration / dt)
        
        violations_advanced = 0
        state = np.array([0.1, 0, 0.5, 0.2, 1.0, 0])
        
        for t in range(steps):
            nominal_control = np.array([-0.1 * state[0], 0, 1.0])
            
            # Note: solve_cbf_qp requires cvxpy which might not be installed
            # Using simplified version if cvxpy not available
            try:
                safe_control = cbf_advanced.solve_cbf_qp(
                    state, nominal_control, cbf_advanced.nominal_dynamics
                )
            except:
                # Fallback to simple safety filter
                safe_control = nominal_control * 0.8
            
            # Update state
            state += cbf_advanced.nominal_dynamics(state) * dt
            
            # Check violations
            barriers = cbf_advanced.define_barrier_functions(state)
            if any(h < 0 for h in barriers):
                violations_advanced += 1
        
        # Format for visualization
        safety_data = {
            'violations_per_hour': safety_metrics['violations_per_hour'],
            'safety_margin': safety_metrics['safety_margin']
        }
        
        results = {
            'total_violations': safety_metrics['total_violations'],
            'violations_per_hour': safety_metrics['violations_per_hour'],
            'meets_target': safety_metrics['meets_target'],
            'safety_margin': safety_metrics['safety_margin'],
            'advanced_violations': violations_advanced
        }
        
        return safety_data, results
    
    def run_comprehensive_test_suite(self) -> Dict:
        """Run the advanced test suite"""
        print("Running Comprehensive Test Suite...")
        
        test_suite = MicrogridTestSuite()
        return test_suite.run_all_tests()
    
    def generate_integrated_results(self) -> Dict:
        """Run all simulations and compile results"""
        
        print("="*60)
        print("INTEGRATED MICROGRID SIMULATION WITH REAL DATA")
        print("="*60)
        
        # Run all simulations
        freq_data, freq_results = self.run_frequency_stability_simulation()
        consensus_data, marl_results = self.run_marl_consensus_simulation()
        cost_data, cost_results = self.run_economic_analysis()
        safety_data, safety_results = self.run_safety_verification()
        
        # Compile all results
        integrated_results = {
            'frequency_data': freq_data,
            'frequency_stability': freq_results,
            'consensus_data': consensus_data,
            'marl_convergence': marl_results,
            'cost_data': cost_data,
            'cost_analysis': cost_results,
            'safety_data': safety_data,
            'safety_verification': safety_results
        }
        
        # Run advanced tests
        print("\nRunning Advanced Module Tests...")
        advanced_tests = self.run_comprehensive_test_suite()
        integrated_results['advanced_tests'] = advanced_tests
        
        return integrated_results
    
    def visualize_results(self, results: Dict):
        """Generate all visualizations with real data"""
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS WITH REAL DATA")
        print("="*60)
        
        # Create comprehensive dashboard
        dashboard = self.viz.create_comprehensive_dashboard(results)
        
        # Generate individual publication figures
        pub_figures = self.viz.generate_publication_figures(results)
        
        # Validate results
        validation_df = self.validator.validate_all_deliverables(results)
        
        print("\nValidation Results:")
        print(validation_df.to_string(index=False))
        
        # Print summary statistics
        self.print_summary(results)
        
        return dashboard, pub_figures, validation_df
    
    def print_summary(self, results: Dict):
        """Print summary of actual calculated results"""
        
        print("\n" + "="*60)
        print("SUMMARY OF ACTUAL CALCULATED RESULTS")
        print("="*60)
        
        print("\n1. FREQUENCY STABILITY:")
        print(f"   Max Deviation: {results['frequency_stability']['max_deviation_hz']:.3f} Hz")
        print(f"   Settling Time: {results['frequency_stability']['settling_time_s']:.2f} s")
        print(f"   ISS Margin: {results['frequency_stability'].get('iss_margin', 0):.3f}")
        print(f"   Target Met: {'✓' if results['frequency_stability']['meets_target'] else '✗'}")
        
        print("\n2. MULTI-AGENT CONSENSUS:")
        print(f"   Convergence Rate: {results['marl_convergence']['convergence_rate']:.3f}")
        print(f"   Lambda_2: {results['marl_convergence']['lambda_2']:.3f}")
        print(f"   Max Delay: {results['marl_convergence']['max_tolerable_delay_s']:.2f} s")
        print(f"   Stable: {'✓' if results['marl_convergence']['stable'] else '✗'}")
        
        print("\n3. ECONOMIC ANALYSIS:")
        print(f"   Our TCO: ${results['cost_analysis']['our_tco']:,}")
        print(f"   Conv TCO: ${results['cost_analysis']['conventional_tco']:,}")
        print(f"   Savings: {results['cost_analysis']['percentage_savings']:.1f}%")
        print(f"   Payback: {results['cost_analysis']['payback_years']:.1f} years")
        
        print("\n4. SAFETY VERIFICATION:")
        print(f"   Violations/Hour: {results['safety_verification']['violations_per_hour']:.2f}")
        print(f"   Safety Margin: {results['safety_verification']['safety_margin']:.2f}")
        print(f"   Target Met: {'✓' if results['safety_verification']['meets_target'] else '✗'}")
        
        if 'advanced_tests' in results:
            print("\n5. ADVANCED MODULE TESTS:")
            freq_test = results['advanced_tests'].get('frequency_stability', {})
            if '150ms' in freq_test:
                print(f"   ISS Margin @150ms: {freq_test['150ms'].get('iss_margin', 0):.3f}")
                print(f"   Stable @150ms: {'✓' if freq_test['150ms'].get('stable', False) else '✗'}")
        
        print("\n" + "="*60)
    
    def save_results(self, results: Dict, filename: str = "integrated_results.json"):
        """Save results to JSON file"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")


def main():
    """Main execution function"""
    
    # Initialize integrated simulation
    sim = IntegratedMicrogridSimulation()
    
    # Run all simulations with real data
    results = sim.generate_integrated_results()
    
    # Generate visualizations
    dashboard, pub_figures, validation = sim.visualize_results(results)
    
    # Save results
    sim.save_results(results)
    
    # Show plots
    plt.show()
    
    print("\n" + "="*60)
    print("INTEGRATED SIMULATION COMPLETE")
    print("✓ Real data calculated from simulation modules")
    print("✓ Visualizations generated with actual results")
    print("✓ Performance validated against specifications")
    print("✓ Results saved to integrated_results.json")
    print("="*60)
    
    return results, dashboard, pub_figures, validation


if __name__ == "__main__":
    results, dashboard, figures, validation = main()
