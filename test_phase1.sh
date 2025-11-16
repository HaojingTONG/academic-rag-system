#!/bin/bash

# Phase 1 功能测试脚本
# 测试新增的API endpoints

API_BASE="http://localhost:8000"

echo "🧪 Testing Phase 1 Enhancements..."
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
total_tests=0
passed_tests=0

# Helper function
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    total_tests=$((total_tests + 1))
    echo -n "Testing $name... "

    status=$(curl -s -o /dev/null -w "%{http_code}" "$url")

    if [ "$status" -eq "$expected_status" ]; then
        echo -e "${GREEN}✓ PASSED${NC} (HTTP $status)"
        passed_tests=$((passed_tests + 1))
    else
        echo -e "${RED}✗ FAILED${NC} (Expected $expected_status, got $status)"
    fi
}

# Test 1: Papers List
echo "📚 Testing Library Panel APIs:"
echo "------------------------------"
test_endpoint "Papers list" "$API_BASE/papers?limit=5"
test_endpoint "Papers search" "$API_BASE/papers?search=transformer"
test_endpoint "Paper detail" "$API_BASE/papers/1706.03762"
echo ""

# Test 2: Suggestions
echo "💡 Testing Smart Suggestions:"
echo "----------------------------"
test_endpoint "Question suggestions" "$API_BASE/papers/suggest/questions?limit=5"
echo ""

# Test 3: Config Management
echo "⚙️  Testing Settings Panel APIs:"
echo "-------------------------------"
test_endpoint "Get config" "$API_BASE/config"
echo ""

# Test 4: Config Update (with data)
echo -n "Testing config update... "
total_tests=$((total_tests + 1))
response=$(curl -s -X PUT "$API_BASE/config" \
    -H "Content-Type: application/json" \
    -d '{"top_k": 10}')

if echo "$response" | grep -q "Configuration updated successfully"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    passed_tests=$((passed_tests + 1))
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "Response: $response"
fi
echo ""

# Test 5: Config Reset
echo -n "Testing config reset... "
total_tests=$((total_tests + 1))
response=$(curl -s -X POST "$API_BASE/config/reset")

if echo "$response" | grep -q "Configuration reset to defaults"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    passed_tests=$((passed_tests + 1))
else
    echo -e "${RED}✗ FAILED${NC}"
    echo "Response: $response"
fi
echo ""

# Test 6: Existing endpoints still work
echo "🔍 Testing Existing APIs (regression):"
echo "--------------------------------------"
test_endpoint "Health check" "$API_BASE/health"
test_endpoint "System stats" "$API_BASE/stats"
echo ""

# Summary
echo "=================================="
echo "📊 Test Summary:"
echo "   Total: $total_tests"
echo -e "   Passed: ${GREEN}$passed_tests${NC}"
echo -e "   Failed: ${RED}$((total_tests - passed_tests))${NC}"
echo ""

if [ "$passed_tests" -eq "$total_tests" ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Check the output above.${NC}"
    exit 1
fi
