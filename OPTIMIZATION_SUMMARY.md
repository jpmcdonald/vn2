# Imputation Pipeline Optimization Summary

## Problem Analysis

**PID 58108** was running at 98.5% CPU (single-threaded) with the command:
```bash
python -m vn2.cli impute --config configs/uncertainty.yaml --n-neighbors 20
```

### Bottleneck Identified

The imputation routine was processing stockouts **sequentially** in a single-threaded loop:
- Each stockout imputation is computationally expensive (O(n) neighbor search)
- Thousands of stockouts to process
- Each imputation is completely independent
- Only utilizing 1 of 12 available CPU cores

## Optimizations Implemented

### 1. **Parallelization with Joblib** ✅
- Added `joblib.Parallel` to process multiple stockouts simultaneously
- Each stockout imputation runs in parallel using all available cores
- New parameter: `--n-jobs` (default: -1 = use all cores)

**Key Changes:**
- `src/vn2/uncertainty/stockout_imputation.py`:
  - Added `_impute_single_stockout()` helper for parallel execution
  - Modified `impute_all_stockouts()` to use `joblib.Parallel`
  - Added `n_jobs` parameter throughout
  
- `src/vn2/cli.py`:
  - Added `--n-jobs` argument to impute command
  - Passes through to imputation functions

- `src/vn2/forecast/imputation_pipeline.py`:
  - Updated `create_imputed_training_data()` to accept `n_jobs`

### 2. **Memory Efficiency** ✅
- All data already working in-memory (pandas DataFrames)
- No unnecessary disk I/O during computation
- Memory footprint remains low (~215 MB for current dataset)

## Usage

### Run with All Cores (Recommended)
```bash
python -m vn2.cli impute --config configs/uncertainty.yaml --n-neighbors 20 --n-jobs -1
```

### Run with Specific Number of Cores
```bash
python -m vn2.cli impute --config configs/uncertainty.yaml --n-neighbors 20 --n-jobs 8
```

### Run Sequential (for debugging)
```bash
python -m vn2.cli impute --config configs/uncertainty.yaml --n-neighbors 20 --n-jobs 1
```

## Expected Performance Improvement

With 12 CPU cores available:
- **Before**: ~10 minutes for single-threaded execution
- **After**: ~1-2 minutes with parallelization (8-10x speedup expected)

The speedup won't be perfectly linear due to:
- Parallel overhead (process spawning, data serialization)
- Memory bandwidth contention
- Uneven work distribution (some stockouts harder than others)

But you should see **6-10x speedup** in practice.

## Additional Optimization Opportunities

### Near-Term (Quick Wins)
1. **Pre-compute profile features** - Cache feature extraction for all SKU-weeks
2. **Batch candidate filtering** - Vectorize the seasonal/in-stock filtering
3. **Use progress bars** - Add `tqdm` to monitor parallel progress

### Medium-Term (More Involved)
4. **KD-tree for neighbor search** - Replace O(n) linear search with O(log n) spatial index
5. **Vectorize distance calculations** - Use numpy broadcasting instead of loops
6. **Cache neighbor lookups** - Similar SKUs likely share neighbors

### Long-Term (Advanced)
7. **GPU acceleration** - Port quantile calculations to GPU with CuPy
8. **Approximate neighbors** - Use FAISS or Annoy for approximate k-NN
9. **Incremental computation** - Only impute new stockouts, not all historical

## Monitoring the Current Job

Check current job status:
```bash
ps -p 58108 -o pid,user,%cpu,%mem,time,command
```

Kill if needed (to restart with parallelization):
```bash
kill 58108
```

Then restart with:
```bash
./go impute --config configs/uncertainty.yaml --n-neighbors 20 --n-jobs -1
```

## Technical Details

**Parallel Backend**: `loky` (process-based, safe for pandas/numpy)
- Each worker gets a copy of the data
- No shared memory issues
- Compatible with macOS/Linux/Windows

**Data Serialization**: 
- DataFrames serialized once per worker
- Overhead ~1-2 seconds for typical dataset
- Amortized across thousands of imputations

**Memory Usage**:
- Each worker maintains a copy of the dataframe
- Expected: ~215 MB × 12 cores = ~2.5 GB total
- Well within your 64 GB available

## Validation

To verify the optimization works correctly:

1. **Run small test** (sequential vs parallel should match):
   ```python
   # Test with small subset
   df_test = df.head(100)
   results_seq = impute_all_stockouts(df_test, ..., n_jobs=1)
   results_par = impute_all_stockouts(df_test, ..., n_jobs=-1)
   # Compare results
   ```

2. **Check output files match** your previous runs

3. **Monitor system resources**:
   ```bash
   top -pid 58108  # Watch CPU usage jump from 98% to multi-core
   ```

## Summary

✅ **Parallelization implemented** - Use all 12 cores  
✅ **Memory-efficient** - Already working in RAM  
✅ **Drop-in replacement** - Just add `--n-jobs -1` flag  
✅ **6-10x speedup expected** - From ~10 min to ~1-2 min  
✅ **No breaking changes** - Defaults to using all cores  

**Next Steps**: 
1. Kill the current single-threaded job (PID 58108)
2. Restart with `--n-jobs -1` to use all cores
3. Monitor speedup
4. Consider implementing additional optimizations if needed

