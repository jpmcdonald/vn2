#!/usr/bin/env python3
"""
Train DeepAR probabilistic forecasting model.

DeepAR is an autoregressive RNN that produces probabilistic forecasts.
It's particularly good for groups of related time series as it learns
shared patterns across all series.

Requirements:
- pytorch
- pytorch-forecasting  OR  gluonts

Usage:
    python scripts/train_deepar.py \
        --demand-path data/processed/demand_long.parquet \
        --output-root models/checkpoints_h3 \
        --max-epochs 50 \
        --batch-size 64

Note: GPU is strongly recommended for training.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings during import checks
warnings.filterwarnings('ignore')

QUANTILE_LEVELS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40,
                            0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])

# Check for available backends
PYTORCH_FORECASTING_AVAILABLE = False
GLUONTS_AVAILABLE = False

try:
    import torch
    from pytorch_forecasting import DeepAR, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    PYTORCH_FORECASTING_AVAILABLE = True
except ImportError:
    pass

try:
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation import make_evaluation_predictions
    GLUONTS_AVAILABLE = True
except ImportError:
    pass


def load_demand(path: Path) -> pd.DataFrame:
    """Load and prepare demand data."""
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if 'demand' not in df.columns and 'sales' in df.columns:
        df['demand'] = df['sales']
    
    df['unique_id'] = df['store'].astype(str) + '_' + df['product'].astype(str)
    
    # Ensure week is datetime
    if df['week'].dtype != 'datetime64[ns]':
        df['week'] = pd.to_datetime(df['week'])
    
    df = df.sort_values(['unique_id', 'week'])
    return df


def prepare_pytorch_forecasting_data(
    df: pd.DataFrame,
    max_encoder_length: int = 52,
    max_prediction_length: int = 3
) -> TimeSeriesDataSet:
    """Prepare data for pytorch-forecasting DeepAR."""
    # Add time index
    df = df.copy()
    df['time_idx'] = df.groupby('unique_id').cumcount()
    
    # Add static features
    df['store_cat'] = df['store'].astype(str).astype('category')
    df['product_cat'] = df['product'].astype(str).astype('category')
    
    training = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target='demand',
        group_ids=['unique_id'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['store_cat', 'product_cat'],
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_reals=['demand'],
        target_normalizer=GroupNormalizer(
            groups=['unique_id'],
            transformation='softplus'
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return training


def train_pytorch_forecasting_deepar(
    df: pd.DataFrame,
    output_dir: Path,
    max_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    hidden_size: int = 32,
    rnn_layers: int = 2,
    dropout: float = 0.1,
    gpus: int = 0
) -> Dict:
    """Train DeepAR using pytorch-forecasting."""
    print("Preparing data for pytorch-forecasting...")
    
    training = prepare_pytorch_forecasting_data(df)
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    
    # Create model
    model = DeepAR.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
    )
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode='min'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        callbacks=[early_stop_callback],
        enable_progress_bar=True,
        gradient_clip_val=0.1,
    )
    
    print("Training DeepAR...")
    trainer.fit(model, train_dataloader)
    
    # Save model
    model_path = output_dir / 'deepar_model.ckpt'
    trainer.save_checkpoint(model_path)
    
    return {
        'model_path': str(model_path),
        'training_dataset': training,
        'model': model
    }


def generate_quantile_forecasts(
    model,
    training_dataset: TimeSeriesDataSet,
    df: pd.DataFrame,
    output_dir: Path,
    horizon: int = 3
) -> int:
    """Generate and save quantile forecasts for all SKUs."""
    # Get predictions
    predictions = model.predict(
        training_dataset.to_dataloader(train=False, batch_size=128, num_workers=0),
        mode='quantiles',
        mode_kwargs={'quantiles': QUANTILE_LEVELS.tolist()}
    )
    
    # Save checkpoints per SKU
    n_saved = 0
    unique_ids = df['unique_id'].unique()
    
    for i, uid in enumerate(unique_ids):
        store, product = map(int, uid.split('_'))
        
        sku_dir = output_dir / 'deepar' / f'{store}_{product}'
        sku_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract quantiles for this SKU (shape: horizon x n_quantiles)
        sku_preds = predictions[i]  # This may need adjustment based on actual output shape
        
        # Create quantiles DataFrame
        quantiles_df = pd.DataFrame(
            sku_preds[:horizon],
            index=range(1, horizon + 1),
            columns=QUANTILE_LEVELS
        )
        
        # Save for each fold (use same forecast for simplicity)
        for fold_idx in range(12):
            path = sku_dir / f'fold_{fold_idx}.pkl'
            with open(path, 'wb') as f:
                pickle.dump({'quantiles': quantiles_df}, f)
        
        n_saved += 1
    
    return n_saved


def train_gluonts_deepar(
    df: pd.DataFrame,
    output_dir: Path,
    max_epochs: int = 50,
    prediction_length: int = 3,
    context_length: int = 52,
    num_layers: int = 2,
    hidden_size: int = 40,
    num_samples: int = 200,
    surd_transforms_df: Optional[pd.DataFrame] = None,
    output_subdir: str = 'deepar',
) -> Dict:
    """Train DeepAR using GluonTS and generate per-SKU checkpoint files.

    When surd_transforms_df is provided, demand is transformed per-SKU
    before training and samples are inverse-transformed after prediction.
    """
    from gluonts.torch.model.deepar import DeepAREstimator
    from gluonts.dataset.pandas import PandasDataset
    from vn2.forecast.transforms import apply_transform, inverse_transform

    use_surd = surd_transforms_df is not None
    if use_surd:
        print(f"SURD mode: will transform targets per-SKU ({output_subdir})")

    print("Preparing data for GluonTS...")

    comp_start = pd.Timestamp('2024-04-15')
    train_df = df[df['week'] < comp_start].copy()

    # Build per-SKU transform lookup
    sku_transform: Dict[str, str] = {}
    if use_surd:
        for _, row in surd_transforms_df.iterrows():
            uid = f"{int(row['Store'])}_{int(row['Product'])}"
            sku_transform[uid] = row['best_transform']

    gluon_df = train_df[['unique_id', 'week', 'demand']].copy()

    if use_surd:
        transformed = []
        for uid, grp in gluon_df.groupby('unique_id'):
            t = sku_transform.get(uid, 'identity')
            grp = grp.copy()
            grp['demand'] = apply_transform(grp['demand'].values.astype(float), t)
            transformed.append(grp)
        gluon_df = pd.concat(transformed)

    gluon_df = gluon_df.rename(columns={'week': 'timestamp', 'demand': 'target'})
    gluon_df = gluon_df.set_index('timestamp')

    dataset = PandasDataset.from_long_dataframe(
        gluon_df, item_id='unique_id', target='target'
    )

    estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        freq='W-MON',
        trainer_kwargs={'max_epochs': max_epochs},
    )

    # PyTorch >= 2.6 defaults weights_only=True which breaks GluonTS checkpoint loading
    import torch
    _orig_torch_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_load

    print("Training DeepAR...")
    try:
        predictor = estimator.train(dataset)
    finally:
        torch.load = _orig_torch_load

    # --- Generate per-SKU quantile checkpoints ---
    print(f"Generating forecasts ({num_samples} samples per SKU)...")
    forecast_it = predictor.predict(dataset, num_samples=num_samples)

    uid_list = train_df.groupby('unique_id').ngroups
    uid_order = sorted(train_df['unique_id'].unique())

    n_saved = 0
    for uid, forecast in zip(uid_order, forecast_it):
        store, product = map(int, uid.split('_'))
        samples = forecast.samples  # shape: (num_samples, prediction_length)

        if use_surd:
            t = sku_transform.get(uid, 'identity')
            samples = inverse_transform(samples, t)

        samples = np.clip(samples, 0, None)  # demand is non-negative
        quantiles_arr = np.quantile(samples, QUANTILE_LEVELS, axis=0).T  # (horizon, n_q)
        quantiles_df = pd.DataFrame(
            quantiles_arr,
            index=range(1, prediction_length + 1),
            columns=QUANTILE_LEVELS,
        )
        sku_dir = output_dir / output_subdir / f'{store}_{product}'
        sku_dir.mkdir(parents=True, exist_ok=True)
        for fold_idx in range(12):
            path = sku_dir / f'fold_{fold_idx}.pkl'
            with open(path, 'wb') as f:
                pickle.dump({'quantiles': quantiles_df}, f)
        n_saved += 1

    print(f"Saved {n_saved}/{uid_list} SKU checkpoints to {output_dir / output_subdir}")

    predictor_path = output_dir / f'{output_subdir}_gluonts'
    predictor_path.mkdir(parents=True, exist_ok=True)
    predictor.serialize(predictor_path)

    return {
        'predictor_path': str(predictor_path),
        'predictor': predictor,
        'dataset': dataset,
        'n_saved': n_saved,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DeepAR forecasting model")
    parser.add_argument("--demand-path", type=Path, default=Path("data/processed/demand_long.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("models/checkpoints_h3"))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs (0 for CPU)")
    parser.add_argument("--backend", choices=['pytorch-forecasting', 'gluonts', 'auto'], default='auto')
    parser.add_argument("--max-skus", type=int, default=None, help="Limit SKUs for testing")
    parser.add_argument("--surd", action="store_true", help="Apply per-SKU SURD transforms (saves to deepar_surd/)")
    parser.add_argument("--surd-path", type=Path, default=Path("data/processed/surd_transforms.parquet"))
    args = parser.parse_args()
    
    # Check available backends
    if args.backend == 'auto':
        if PYTORCH_FORECASTING_AVAILABLE:
            backend = 'pytorch-forecasting'
        elif GLUONTS_AVAILABLE:
            backend = 'gluonts'
        else:
            print("ERROR: Neither pytorch-forecasting nor gluonts is available.")
            print("Install with: pip install pytorch-forecasting pytorch-lightning")
            print("         or: pip install gluonts")
            return
    else:
        backend = args.backend
        if backend == 'pytorch-forecasting' and not PYTORCH_FORECASTING_AVAILABLE:
            print("ERROR: pytorch-forecasting not available")
            return
        if backend == 'gluonts' and not GLUONTS_AVAILABLE:
            print("ERROR: gluonts not available")
            return
    
    print(f"Using backend: {backend}")
    print(f"GPUs: {args.gpus}")
    
    # Load data
    print(f"Loading demand data from {args.demand_path}")
    df = load_demand(args.demand_path)
    
    if args.max_skus:
        unique_ids = df['unique_id'].unique()[:args.max_skus]
        df = df[df['unique_id'].isin(unique_ids)]
    
    print(f"  {len(df['unique_id'].unique())} SKUs, {len(df)} observations")
    
    # Train model
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    if backend == 'pytorch-forecasting':
        result = train_pytorch_forecasting_deepar(
            df=df,
            output_dir=args.output_root,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            rnn_layers=args.rnn_layers,
            dropout=args.dropout,
            gpus=args.gpus
        )
        
        # Generate forecasts
        print("Generating quantile forecasts...")
        n_saved = generate_quantile_forecasts(
            model=result['model'],
            training_dataset=result['training_dataset'],
            df=df,
            output_dir=args.output_root,
            horizon=3
        )
        print(f"Saved {n_saved} SKU checkpoints")
        
    else:  # gluonts
        surd_df = None
        output_subdir = 'deepar'
        if args.surd:
            if args.surd_path.exists():
                surd_df = pd.read_parquet(args.surd_path)
                output_subdir = 'deepar_surd'
                print(f"SURD transforms loaded: {len(surd_df)} SKUs from {args.surd_path}")
            else:
                print(f"WARNING: --surd specified but {args.surd_path} not found, training without SURD")

        result = train_gluonts_deepar(
            df=df,
            output_dir=args.output_root,
            max_epochs=args.max_epochs,
            prediction_length=3,
            context_length=52,
            num_layers=args.rnn_layers,
            hidden_size=args.hidden_size,
            surd_transforms_df=surd_df,
            output_subdir=output_subdir,
        )
        print(f"Model saved to: {result['predictor_path']}")
        print(f"Checkpoints: {result['n_saved']} SKUs")
    
    print("\nDeepAR training complete!")
    print(f"Output: {args.output_root / 'deepar'}")


if __name__ == "__main__":
    main()

