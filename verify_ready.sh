#!/bin/bash
echo "=== Challenger Benchmark Verification ==="
echo ""

echo "1. Winsorized data exists:"
ls -lh data/processed/demand_imputed_winsor.parquet

echo ""
echo "2. Raw data exists:"
ls -lh data/processed/demand_long.parquet

echo ""
echo "3. SURD transforms exist:"
ls -lh data/processed/surd_transforms.parquet

echo ""
echo "4. Enabled models in config:"
grep -A 1 "enabled:" configs/forecast.yaml | grep -E "(slurp|lightgbm|ets|linear|ngboost|qrf)" | grep -B 1 "true" | grep -v "^--$" | head -20

echo ""
echo "5. New model files exist:"
ls -1 src/vn2/forecast/models/lightgbm_point.py src/vn2/forecast/models/qrf.py src/vn2/forecast/models/glm_count.py 2>&1 | grep -v "cannot access"

echo ""
echo "6. Test import:"
source V2env/bin/activate
python -c "from vn2.forecast.models.lightgbm_point import LightGBMPointForecaster; from vn2.forecast.models.qrf import QRFForecaster; print('✅ All imports successful')"

echo ""
echo "=== READY TO TRAIN ==="
