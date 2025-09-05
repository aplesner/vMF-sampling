try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    raise ImportError("wandb is not installed. Please install wandb to use logging features.")

from .config import VMFConfig


class VMFLogger:
    """Logger for vMF sampling experiments with CSV output and wandb compatibility."""
    
    def __init__(self, config: VMFConfig):
        """
        Initialize logger.
        
        Parameters
        ----------
        config : VMFConfig
        """
        self.config = config

        # Initialize wandb if specified and API key is available
        if self.config.wandb_project and _WANDB_AVAILABLE:
            self._init_wandb()
        else:
            raise ImportError("wandb is not available. Please install wandb to use this logger.")
 

    def _init_wandb(self):
        """Initialize wandb with proper configuration."""
        if _WANDB_AVAILABLE:
            self.run = wandb.init(
                project=self.config.wandb_project,
                mode='offline' if self.config.wandb_offline else 'online', 
                config=self.config.to_dict()
            )
            wandb.Settings(quiet=True)
            print(f"wandb initialized for project: {self.config.wandb_project}")


    def log_experiment(self, config: VMFConfig, timing_results: dict[str, float | str]):
        """
        Log experiment results.
        
        Parameters
        ----------
        config : VMFConfig
            Experiment configuration.
        timing_results : dict
            Dictionary with timing results.
        """
        result = {
            'dimension': config.mu_dim,  # Use 'dimension' for clearer plotting
            'mean_runtime': timing_results.get('mean_time', None),  # Explicit mean runtime
            'median_runtime': timing_results.get('median_time', None),
            'std_runtime': timing_results.get('std', None),
            'kappa': config.kappa,
            'num_samples': config.num_samples,
            'seed': config.seed,
            **{k: v for k, v in timing_results.items() if k not in ['mean_time', 'median_time', 'std', 'device']}
        }
        
        self.run.log(result)
            
    
    def finish(self):
        """Clean up and finish logging."""
        self.run.finish()
