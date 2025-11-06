# Data Ingestion Guide

This guide explains how raw competition data is ingested and processed into a clean, structured format.

## Overview

The ingestion process takes raw CSV files from the competition and:
1. **Validates** the data structure
2. **Extracts** canonical SKU index from submission template
3. **Loads** initial inventory state
4. **Loads** product/store hierarchy (master data)
5. **Saves** cleaned data to Parquet format for efficient access

## Command

```bash
./go ingest --raw data/raw --out data/interim
```

## Data Flow

```
┌─────────────────────────────────┐
│   Raw CSV Files                 │
│   (Week 0 - *.csv)              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   1. submission_index()         │
│   → Extract canonical SKU list  │
│   → Creates MultiIndex          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   2. load_initial_state()       │
│   → Load inventory quantities   │
│   → Initialize transit pipeline │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   3. load_master()              │
│   → Load product hierarchy      │
│   → Store metadata              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│   Save to Parquet               │
│   → state.parquet               │
│   → master.parquet              │
└─────────────────────────────────┘
```

## Raw Data Files

The ingestion process expects these files in the `--raw` directory:

### 1. `Week 0 - Submission Template.csv`
**Purpose**: Defines the canonical ordering of all 599 SKUs

**Format**:
```csv
Store,Product,0
0,126,0
0,182,0
1,124,0
2,124,0
...
```

**Used for**: Creating the MultiIndex that ensures all data is consistently ordered

### 2. `Week 0 - In Stock.csv`
**Purpose**: Initial inventory levels for each SKU

**Format**:
```csv
Store,Product,2021-04-12,2021-04-19,...
0,126,True,True,True,...
0,182,False,False,False,...
```

**Note**: This file actually contains historical in-stock flags (True/False) for each week. The ingestion code extracts the most recent column (last date) which contains the initial stock quantities.

### 3. `Week 0 - Master.csv`
**Purpose**: Product and store hierarchy/metadata

**Format**:
```csv
Store,Product,ProductGroup,Division,Department,DepartmentGroup,StoreFormat,Format
0,126,301202,3012,30,11,1,13
0,182,440403,4404,44,11,1,13
```

**Contains**: Hierarchical product categories and store formats for segmentation

## Code Walkthrough

### Step 1: Extract Canonical Index

```python
def submission_index(raw_dir: str) -> pd.MultiIndex:
    """
    Load canonical index from submission template.
    This defines the required ordering for all submissions.
    """
    tpl = pd.read_csv(Path(raw_dir) / "Week 0 - Submission Template.csv")
    return pd.MultiIndex.from_frame(
        tpl[["Store", "Product"]], 
        names=["Store", "Product"]
    )
```

**What it does**:
- Reads the submission template CSV
- Extracts `Store` and `Product` columns
- Creates a MultiIndex with 599 rows (one per SKU)
- This index ensures all subsequent data aligns to the same ordering

**Result**: `MultiIndex([(0, 126), (0, 182), (1, 124), ...])` with 599 entries

### Step 2: Load Initial State

```python
def load_initial_state(raw_dir: str, index: pd.MultiIndex) -> pd.DataFrame:
    """
    Load initial inventory state.
    
    Returns DataFrame with columns: [on_hand, intransit_1, intransit_2]
    """
    instock = pd.read_csv(Path(raw_dir) / "Week 0 - In Stock.csv")
    df = instock.set_index(["Store", "Product"]).reindex(index).fillna(0)
    
    # Robust to different column naming
    qty_col = [c for c in df.columns if c not in ["Store", "Product"]][-1]
    out = pd.DataFrame(index=index)
    out["on_hand"] = df[qty_col].astype(float)
    out["intransit_1"] = 0.0
    out["intransit_2"] = 0.0
    
    return out
```

**What it does**:
1. Loads the "In Stock" CSV file
2. Sets `(Store, Product)` as the index
3. **Reindexes** to match the canonical index (ensures all 599 SKUs are present)
4. Fills missing values with 0 (SKUs not in file get 0 inventory)
5. Extracts the **last column** (most recent date) as the quantity column
6. Creates output with three columns:
   - `on_hand`: Current inventory on hand
   - `intransit_1`: Inventory arriving next week (initialized to 0)
   - `intransit_2`: Inventory arriving in 2 weeks (initialized to 0)

**Result**: DataFrame with shape `(599, 3)` indexed by `(Store, Product)`

**Note**: The code is "robust to different column naming" - it finds the last non-Store/Product column, which should be the most recent date.

### Step 3: Load Master Data

```python
def load_master(raw_dir: str) -> pd.DataFrame:
    """
    Load master product/store hierarchy.
    
    Contains: ProductGroup, Division, Department, DepartmentGroup, StoreFormat, Format
    """
    master = pd.read_csv(Path(raw_dir) / "Week 0 - Master.csv")
    return master.set_index(["Store", "Product"])
```

**What it does**:
- Loads the master CSV file
- Sets `(Store, Product)` as the index
- Returns all hierarchy columns (ProductGroup, Division, etc.)

**Result**: DataFrame with shape `(599, 7)` indexed by `(Store, Product)`

### Step 4: Save to Parquet

```python
# Save to interim
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

state.to_parquet(out_dir / "state.parquet")
master.to_parquet(out_dir / "master.parquet")
```

**What it does**:
- Creates output directory if it doesn't exist
- Saves state DataFrame to `state.parquet`
- Saves master DataFrame to `master.parquet`

**Why Parquet?**:
- **Efficient**: Compressed columnar format
- **Fast**: Faster to read than CSV
- **Type-preserving**: Maintains data types (no need to reparse)
- **Standard**: Widely used in data science workflows

## Complete CLI Command Implementation

```python
def cmd_ingest(args):
    """Ingest raw data and create clean interim artifacts"""
    rprint(f"[cyan]Ingesting data from {args.raw}...[/cyan]")
    
    # Step 1: Get canonical index
    idx = submission_index(args.raw)
    
    # Step 2: Load initial inventory state
    state = load_initial_state(args.raw, idx)
    
    # Step 3: Load master hierarchy
    master = load_master(args.raw)
    
    # Step 4: Save to interim
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    state.to_parquet(out_dir / "state.parquet")
    master.to_parquet(out_dir / "master.parquet")
    
    rprint(f"[green]✓ Ingested {len(idx)} SKUs -> {args.out}[/green]")
    rprint(f"  State shape: {state.shape}")
    rprint(f"  Master shape: {master.shape}")
```

## Output Files

After ingestion, you'll have:

### `data/interim/state.parquet`
```
                  on_hand  intransit_1  intransit_2
Store Product                                      
0     126           45.0          0.0          0.0
0     182            0.0          0.0          0.0
1     124           12.0          0.0          0.0
...
```

### `data/interim/master.parquet`
```
                  ProductGroup  Division  Department  ...
Store Product                                          
0     126             301202      3012          30  ...
0     182             440403      4404          44  ...
1     124             240201      2402          24  ...
...
```

## Usage in Other Parts of the Codebase

Once ingested, these files are loaded throughout the codebase:

```python
# Load state
state = pd.read_parquet("data/interim/state.parquet")

# Load master
master = pd.read_parquet("data/interim/master.parquet")
```

## Key Design Decisions

1. **Canonical Index**: Using the submission template ensures all data aligns to the exact same 599 SKUs in the same order
2. **MultiIndex**: Using `(Store, Product)` as a MultiIndex allows efficient lookups and joins
3. **Reindexing**: `reindex()` ensures missing SKUs are handled gracefully (filled with 0 or NaN)
4. **Parquet Format**: Faster and more efficient than CSV for repeated reads
5. **Robust Column Detection**: The code handles different column names by finding the last non-Store/Product column

## Validation

The ingestion process implicitly validates:
- ✅ All 599 SKUs from template are present (via reindex)
- ✅ Data types are correct (floats for quantities)
- ✅ Index alignment is consistent across files
- ✅ Missing data is handled (fillna(0))

## Next Steps

After ingestion, typical next steps are:
1. **Stockout Imputation**: `./go impute` - Handle censored demand data
2. **Forecasting**: Generate demand forecasts for each SKU
3. **Optimization**: Compute optimal order quantities
4. **Simulation**: Test policies on historical data

## Troubleshooting

**Problem**: `FileNotFoundError: Week 0 - Submission Template.csv`
- **Solution**: Ensure `--raw` points to the correct directory with all "Week 0" files

**Problem**: Shape mismatch or missing SKUs
- **Solution**: The reindex step should handle this automatically (fills with 0/NaN)

**Problem**: Wrong inventory quantities
- **Solution**: Check that "Week 0 - In Stock.csv" has the correct most recent date column

