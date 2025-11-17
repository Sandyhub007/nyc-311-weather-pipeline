#!/bin/bash
# ML Pipeline Validation Script

echo "============================================================"
echo "🔍 NYC 311 ML PIPELINE VALIDATION"
echo "============================================================"
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service
check_service() {
    local service=$1
    echo -n "Checking $service... "
    if docker compose ps $service | grep -q "running\|healthy"; then
        echo -e "${GREEN}✅ Running${NC}"
        return 0
    else
        echo -e "${RED}❌ Not running${NC}"
        ERRORS=$((ERRORS+1))
        return 1
    fi
}

# Function to run command and check
run_check() {
    local description=$1
    local command=$2
    local expected=$3
    
    echo -n "$description... "
    result=$(eval $command 2>&1)
    
    if echo "$result" | grep -q "$expected"; then
        echo -e "${GREEN}✅ Pass${NC}"
        return 0
    else
        echo -e "${RED}❌ Fail${NC}"
        echo "   Expected: $expected"
        echo "   Got: $result"
        ERRORS=$((ERRORS+1))
        return 1
    fi
}

echo "📋 Step 1: Checking Docker Services"
echo "------------------------------------------------------------"
check_service "postgres"
check_service "redis"
check_service "airflow-scheduler"
check_service "airflow-webserver"
check_service "metabase"
check_service "ml-worker"
echo ""

echo "📋 Step 2: Checking Python Environment"
echo "------------------------------------------------------------"
run_check "Testing ML library imports" \
    "docker compose exec -T ml-worker python -c 'import sklearn, xgboost, prophet, pyspark; print(\"OK\")'" \
    "OK"
echo ""

echo "📋 Step 3: Checking Database"
echo "------------------------------------------------------------"
run_check "Checking data availability" \
    "docker compose exec -T postgres psql -U airflow -d airflow -t -c 'SELECT COUNT(*) FROM nyc_311_bronx_full_year'" \
    "[0-9]"
echo ""

echo "📋 Step 4: Checking File Structure"
echo "------------------------------------------------------------"
echo -n "Checking models directory... "
if [ -d "models" ]; then
    echo -e "${GREEN}✅ Exists${NC}"
else
    echo -e "${YELLOW}⚠️  Missing${NC}"
    WARNINGS=$((WARNINGS+1))
fi

echo -n "Checking reports directory... "
if [ -d "reports" ]; then
    echo -e "${GREEN}✅ Exists${NC}"
else
    echo -e "${YELLOW}⚠️  Missing${NC}"
    WARNINGS=$((WARNINGS+1))
fi

echo -n "Checking ML scripts... "
if [ -f "scripts/ml_classification.py" ] && \
   [ -f "scripts/ml_forecasting.py" ] && \
   [ -f "scripts/spark_regression.py" ] && \
   [ -f "scripts/ml_pipeline.py" ]; then
    echo -e "${GREEN}✅ All present${NC}"
else
    echo -e "${RED}❌ Missing scripts${NC}"
    ERRORS=$((ERRORS+1))
fi
echo ""

echo "📋 Step 5: Checking Network Connectivity"
echo "------------------------------------------------------------"
run_check "Airflow UI" \
    "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080" \
    "200"

run_check "Metabase UI" \
    "curl -s -o /dev/null -w '%{http_code}' http://localhost:3000" \
    "200"
echo ""

echo "📋 Step 6: Quick ML Test (Optional - may take a few minutes)"
echo "------------------------------------------------------------"
echo -n "Run ML classification test? (y/n): "
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Running ML classification test..."
    if docker compose exec -T ml-worker python /app/scripts/ml_classification.py 2>&1 | tail -5 | grep -q "COMPLETED SUCCESSFULLY"; then
        echo -e "${GREEN}✅ ML Classification test passed${NC}"
    else
        echo -e "${RED}❌ ML Classification test failed${NC}"
        echo "   Check logs: docker compose logs ml-worker"
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${YELLOW}⚠️  Skipped ML test${NC}"
    WARNINGS=$((WARNINGS+1))
fi
echo ""

echo "============================================================"
echo "📊 VALIDATION SUMMARY"
echo "============================================================"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Your ML pipeline is fully operational and ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Run complete pipeline: docker compose exec ml-worker python /app/scripts/ml_pipeline.py"
    echo "  2. Access Airflow: http://localhost:8080"
    echo "  3. Access Metabase: http://localhost:3000"
    echo "  4. View models: ls -lh models/"
    echo "  5. View reports: ls -lh reports/"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}✅ VALIDATION PASSED WITH WARNINGS${NC}"
    echo ""
    echo "Warnings: $WARNINGS"
    echo "Your ML pipeline should work, but review warnings above."
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo ""
    echo "Errors: $ERRORS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo ""
    echo "Common fixes:"
    echo "  1. docker compose down && docker compose up -d"
    echo "  2. docker compose build ml-worker"
    echo "  3. Check logs: docker compose logs ml-worker"
    echo "  4. Review: ML_SETUP_GUIDE.md"
    exit 1
fi

