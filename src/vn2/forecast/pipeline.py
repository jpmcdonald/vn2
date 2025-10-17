"""
Forecast training pipeline with checkpoint/resume and parallel execution.
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import warnings

from .models.base import BaseForecaster
from .features import prepare_train_test_split
from .evaluation import evaluate_forecast


@dataclass
class TaskConfig:
    """Configuration for a single forecast task"""
    model_name: str
    sku_id: Tuple[int, int]
    fold_idx: int
    store: int
    product: int


class ForecastPipeline:
    """
    Orchestrates parallel training of forecast models with checkpointing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.models_dir = Path(config['paths']['models'])
        self.progress_file = self.checkpoint_dir / 'progress.json'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracker
        self.progress = self._load_progress()
        
    def _load_progress(self) -> Dict:
        """Load progress from checkpoint"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed': [], 'failed': [], 'data_hash': None}
    
    def _save_progress(self):
        """Save progress to checkpoint"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of data for cache invalidation"""
        # Hash based on shape and first/last rows
        hash_input = f"{df.shape}_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _is_complete(self, task: TaskConfig) -> bool:
        """Check if task already completed"""
        task_key = f"{task.model_name}_{task.store}_{task.product}_{task.fold_idx}"
        return task_key in self.progress['completed']
    
    def _mark_complete(self, task: TaskConfig, status: str = 'success'):
        """Mark task as complete"""
        task_key = f"{task.model_name}_{task.store}_{task.product}_{task.fold_idx}"
        if status == 'success':
            self.progress['completed'].append(task_key)
        else:
            self.progress['failed'].append(task_key)
    
    def generate_tasks(
        self, 
        df: pd.DataFrame,
        models: List[str],
        pilot_skus: Optional[List[Tuple[int, int]]] = None
    ) -> List[TaskConfig]:
        """
        Generate all forecast tasks.
        
        Args:
            df: Full demand dataframe
            models: List of model names to train
            pilot_skus: If provided, only train on these SKUs
            
        Returns:
            List of TaskConfig objects
        """
        # Get unique SKUs
        if pilot_skus is not None:
            skus = pilot_skus
        else:
            skus = df[['Store', 'Product']].drop_duplicates().values.tolist()
            skus = [tuple(sku) for sku in skus]
        
        # Generate all combinations
        tasks = []
        for model_name in models:
            for sku in skus:
                for fold_idx in range(self.config['rolling_origins']):
                    task = TaskConfig(
                        model_name=model_name,
                        sku_id=sku,
                        fold_idx=fold_idx,
                        store=sku[0],
                        product=sku[1]
                    )
                    
                    # Skip if already done
                    if not self._is_complete(task):
                        tasks.append(task)
        
        return tasks
    
    def train_one(
        self,
        task: TaskConfig,
        df: pd.DataFrame,
        master_df: Optional[pd.DataFrame],
        model_factory: callable,
        timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Train a single forecast model.
        
        Args:
            task: Task configuration
            df: Full demand dataframe
            master_df: Master data for hierarchy
            model_factory: Function that creates model instance
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        try:
            # Prepare data
            y_train, X_train, y_test, X_test = prepare_train_test_split(
                df,
                task.sku_id,
                holdout_weeks=self.config['holdout_weeks'],
                fold_idx=task.fold_idx,
                master_df=master_df
            )
            
            if y_train is None or len(y_train) < 13:
                return {
                    'status': 'insufficient_data',
                    'task': asdict(task),
                    'elapsed': time.time() - start_time
                }
            
            # Create and fit model
            model = model_factory()
            model.set_metadata('sku_id', task.sku_id)
            model.set_metadata('fold_idx', task.fold_idx)
            
            # Fit with timeout protection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(y_train, X_train)
            
            # Check timeout
            if time.time() - start_time > timeout:
                return {
                    'status': 'timeout',
                    'task': asdict(task),
                    'elapsed': time.time() - start_time
                }
            
            # Predict
            quantiles_df = model.predict_quantiles(steps=2, X_future=X_test)
            
            # Evaluate
            y_test_vals = y_test.values if len(y_test) == 2 else np.pad(y_test.values, (0, 2 - len(y_test)), constant_values=0)
            metrics = evaluate_forecast(y_test_vals, quantiles_df, include_cost=False)
            
            # Save checkpoint
            checkpoint_path = self._get_checkpoint_path(task)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'quantiles': quantiles_df,
                    'metrics': metrics,
                    'task': asdict(task)
                }, f)
            
            elapsed = time.time() - start_time
            
            return {
                'status': 'success',
                'task': asdict(task),
                'metrics': metrics,
                'elapsed': elapsed
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'task': asdict(task),
                'error': str(e),
                'elapsed': time.time() - start_time
            }
    
    def _get_checkpoint_path(self, task: TaskConfig) -> Path:
        """Get checkpoint file path for a task"""
        return self.checkpoint_dir / task.model_name / f"{task.store}_{task.product}" / f"fold_{task.fold_idx}.pkl"
    
    def train_all(
        self,
        df: pd.DataFrame,
        models: Dict[str, callable],
        master_df: Optional[pd.DataFrame] = None,
        n_jobs: int = 11,
        pilot_skus: Optional[List[Tuple[int, int]]] = None
    ) -> pd.DataFrame:
        """
        Train all models on all SKUs with parallel execution.
        
        Args:
            df: Full demand dataframe
            models: Dictionary mapping model_name -> factory function
            master_df: Master data
            n_jobs: Number of parallel workers
            pilot_skus: If provided, only train on these SKUs
            
        Returns:
            DataFrame with all results
        """
        # Compute data hash
        data_hash = self._compute_data_hash(df)
        if self.progress['data_hash'] != data_hash:
            print(f"Data changed, resetting progress (old: {self.progress['data_hash']}, new: {data_hash})")
            self.progress = {'completed': [], 'failed': [], 'data_hash': data_hash}
            self._save_progress()
        
        # Generate tasks
        model_names = list(models.keys())
        tasks = self.generate_tasks(df, model_names, pilot_skus)
        
        print(f"Total tasks: {len(tasks)}")
        print(f"Already completed: {len(self.progress['completed'])}")
        print(f"To run: {len(tasks)}")
        
        if len(tasks) == 0:
            print("All tasks already complete!")
            return self._load_results()
        
        # Parallel execution with threading backend (avoids system permission issues)
        print(f"Starting parallel execution with {n_jobs} workers (threading backend)...")
        
        results = []
        batch_size = 100
        
        for batch_start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[batch_start:batch_start + batch_size]
            
            batch_results = Parallel(n_jobs=n_jobs, backend='threading', verbose=5)(
                delayed(self.train_one)(
                    task,
                    df,
                    master_df,
                    models[task.model_name],
                    timeout=self.config.get('timeout_per_fit', 120)
                ) for task in batch_tasks
            )
            
            # Update progress
            for result in batch_results:
                task_dict = result['task']
                task = TaskConfig(**task_dict)
                self._mark_complete(task, result['status'])
                results.append(result)
            
            # Save progress after each batch
            self._save_progress()
            print(f"Completed {len(self.progress['completed'])} / {len(self.progress['completed']) + len(tasks)} tasks")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = self.models_dir.parent / 'results' / 'training_results.parquet'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(results_path)
        
        print(f"\nTraining complete!")
        print(f"  Success: {(results_df['status'] == 'success').sum()}")
        print(f"  Failed: {(results_df['status'] == 'failed').sum()}")
        print(f"  Timeout: {(results_df['status'] == 'timeout').sum()}")
        print(f"  Insufficient data: {(results_df['status'] == 'insufficient_data').sum()}")
        
        return results_df
    
    def _load_results(self) -> pd.DataFrame:
        """Load previously saved results"""
        results_path = self.models_dir.parent / 'results' / 'training_results.parquet'
        if results_path.exists():
            return pd.read_parquet(results_path)
        return pd.DataFrame()

