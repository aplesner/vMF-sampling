"""
Logging utilities with CSV output and wandb integration.
"""

import csv
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .config import VMFConfig


class VMFLogger:
    """Logger for vMF sampling experiments with CSV output and wandb compatibility."""
    
    def __init__(self, output_file: str, wandb_project: Optional[str] = None, 
                 wandb_entity: Optional[str] = None):
        """
        Initialize logger.
        
        Parameters
        ----------
        output_file : str
            Path to CSV output file.
        wandb_project : str or None
            wandb project name. If None, only CSV logging is used.
        wandb_entity : str or None
            wandb entity/team name.
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.results = []
        
        # Initialize wandb if specified and API key is available
        self.use_wandb = False
        if wandb_project and self._check_wandb_available():
            self._init_wandb()
    
    def _check_wandb_available(self) -> bool:
        """Check if wandb is available."""
        try:
            import wandb
            return True
        except ImportError:
            print("wandb not installed, using CSV only")
            return False
    
    def _init_wandb(self):
        """Initialize wandb with proper configuration."""
        try:
            import wandb
            
            # Get configuration from environment
            wandb_config = {
                'project': self.wandb_project,
                'entity': self.wandb_entity,
                'mode': os.getenv('WANDB_MODE', 'online'),
            }
            
            # Remove None values
            wandb_config = {k: v for k, v in wandb_config.items() if v is not None}
            
            wandb.init(**wandb_config)
            self.use_wandb = True
            print(f"wandb initialized for project: {self.wandb_project}")
            
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def log_experiment(self, config: VMFConfig, timing_results: Dict[str, float], 
                      profile_results: Optional[Dict[str, Any]] = None):
        """
        Log experiment results.
        
        Parameters
        ----------
        config : VMFConfig
            Experiment configuration.
        timing_results : dict
            Dictionary with timing results.
        profile_results : dict or None
            Profiling results if available.
        """
        result = {
            'dimension': config.mu_dim,  # Use 'dimension' for clearer plotting
            'implementation': config.implementation.value,
            'mean_runtime': timing_results.get('mean_time', None),  # Explicit mean runtime
            'median_runtime': timing_results.get('median_time', None),
            'std_runtime': timing_results.get('std', None),
            'device': timing_results.get('device', config.device),
            'kappa': config.kappa,
            'num_samples': config.num_samples,
            'seed': config.seed,
            **{k: v for k, v in timing_results.items() if k not in ['mean_time', 'median_time', 'std', 'device']}
        }
        
        # Add profiling results if available
        if profile_results:
            # Flatten profiling results for CSV
            result.update(self._flatten_profile_results(profile_results))
        
        self.results.append(result)
        
        # Log to wandb if enabled
        if self.use_wandb:
            try:
                import wandb
                
                # Create wandb-friendly log entry with clear metrics for plotting
                wandb_log = dict(result)
                
                # Add step information for proper time series plotting
                wandb_log['step'] = result['dimension']  # Use dimension as step for x-axis
                
                # Add configuration as wandb config (allow changes for sweeps)
                wandb.config.update(config.to_dict(), allow_val_change=True)
                
                # Add profiling details to wandb if available
                if profile_results and 'top_functions' in profile_results:
                    # Log top functions as a table
                    wandb_log['profiling_table'] = wandb.Table(
                        columns=['function', 'total_time', 'num_calls', 'percentage'],
                        data=[[f['name'], f['total_time'], f['num_calls'], f['percentage']] 
                              for f in profile_results['top_functions'][:5]]
                    )
                
                wandb.log(wandb_log)
                
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
    
    def _flatten_profile_results(self, profile_results: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten complex profiling results for CSV storage."""
        flattened = {}
        
        # Add scalar values
        for key in ['total_time', 'total_calls', 'sampling_time', 'rotation_time', 'other_time']:
            if key in profile_results:
                flattened[f'profile_{key}'] = profile_results[key]
        
        # Add time breakdown percentages
        if 'time_breakdown' in profile_results:
            for key, value in profile_results['time_breakdown'].items():
                flattened[f'profile_{key}'] = value
        
        # Add top functions summary
        if 'top_functions' in profile_results:
            top_funcs = profile_results['top_functions'][:3]  # Top 3 for CSV
            for i, func in enumerate(top_funcs):
                flattened[f'top_func_{i+1}_name'] = func['name'].split('(')[0]  # Function name only
                flattened[f'top_func_{i+1}_time'] = func['total_time']
                flattened[f'top_func_{i+1}_percentage'] = func['percentage']
        
        return flattened
    
    def save_csv(self, force_save=False):
        """Save results to CSV file."""
        if not self.results:
            return
        
        # Check if we should save CSV - either forced or not in wandb-only mode
        if not force_save and hasattr(self, '_offline_mode') and not self._offline_mode:
            return
            
        fieldnames = list(self.results[0].keys())
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results saved to {self.output_file}")
        
    def set_offline_mode(self, offline: bool):
        """Set whether to save CSV files locally."""
        self._offline_mode = offline
    
    def finish(self):
        """Clean up and finish logging."""
        self.save_csv()
        
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                print(f"Error finishing wandb: {e}")