#!/bin/bash
# Check if SIP evaluation prerequisites are met

echo "============================================================"
echo "SIP Evaluation Readiness Check"
echo "============================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

READY=true

# 1. Check demand data
echo "1. Checking demand data..."
if [ -f "data/processed/demand_imputed_capped.parquet" ]; then
    SIZE=$(du -h data/processed/demand_imputed_capped.parquet | cut -f1)
    echo -e "   ${GREEN}✓${NC} demand_imputed_capped.parquet exists (${SIZE})"
else
    echo -e "   ${RED}✗${NC} demand_imputed_capped.parquet NOT FOUND"
    READY=false
fi

# 2. Check state data
echo "2. Checking initial state data..."
if [ -f "data/interim/state.parquet" ]; then
    SIZE=$(du -h data/interim/state.parquet | cut -f1)
    echo -e "   ${GREEN}✓${NC} state.parquet exists (${SIZE})"
else
    echo -e "   ${YELLOW}⚠${NC} state.parquet NOT FOUND (will use zero initial state)"
fi

# 3. Check trained models
echo "3. Checking trained models..."
MODELS=("slurp_stockout_aware" "slurp_bootstrap" "seasonal_naive")
for MODEL in "${MODELS[@]}"; do
    if [ -d "models/checkpoints/$MODEL" ]; then
        COUNT=$(find "models/checkpoints/$MODEL" -name "fold_*.pkl" | wc -l | tr -d ' ')
        echo -e "   ${GREEN}✓${NC} $MODEL: $COUNT checkpoints"
    else
        echo -e "   ${RED}✗${NC} $MODEL: NOT FOUND"
        READY=false
    fi
done

# 4. Check Python environment
echo "4. Checking Python environment..."
if [ -d "V2env" ]; then
    echo -e "   ${GREEN}✓${NC} Virtual environment exists"
    
    # Check if we can import required modules
    source V2env/bin/activate 2>/dev/null
    if python -c "import vn2.analyze.sip_opt" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} SIP module importable"
    else
        echo -e "   ${RED}✗${NC} Cannot import vn2.analyze.sip_opt"
        READY=false
    fi
else
    echo -e "   ${RED}✗${NC} Virtual environment NOT FOUND"
    READY=false
fi

# 5. Check disk space
echo "5. Checking disk space..."
AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
echo -e "   ${GREEN}✓${NC} Available space: $AVAILABLE"

# 6. Check CPU cores
echo "6. Checking CPU resources..."
if command -v sysctl &> /dev/null; then
    CORES=$(sysctl -n hw.logicalcpu)
    echo -e "   ${GREEN}✓${NC} Logical CPU cores: $CORES"
else
    echo -e "   ${YELLOW}⚠${NC} Cannot determine CPU count"
fi

# Summary
echo ""
echo "============================================================"
if [ "$READY" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - READY TO RUN${NC}"
    echo ""
    echo "Run evaluation with:"
    echo "  source activate.sh"
    echo "  python -m vn2.cli eval-models --use-sip-optimization --out-suffix v4 --n-jobs 12 --resume"
else
    echo -e "${RED}❌ SOME CHECKS FAILED - FIX ISSUES BEFORE RUNNING${NC}"
fi
echo "============================================================"

