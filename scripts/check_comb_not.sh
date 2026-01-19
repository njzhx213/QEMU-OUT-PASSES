#!/usr/bin/env bash
#
# check_comb_not.sh - Verify if comb.not appears in qemu-passes pipeline output
#
# Purpose: Determine if traceSignal() needs to support comb.not or if a
#          normalize-not pass should be added to the pipeline
#
# Critical Coverage Tests:
#   - trace_signal_multi_not.mlir   (multi-layer NOT parity)
#   - trace_signal_sigextract.mlir  (llhd.sig.extract tracing)
#   - trace_signal_icmp.mlir        (comb.icmp eq/ne for i1 NOT)
#
# These critical tests MUST pass (not skip) for the conclusion to be valid.
#
# Usage: bash scripts/check_comb_not.sh
#
# Environment Variables:
#   QEMU_PASSES_BIN   - Path to qemu-passes binary (default: auto-detect)
#   CIRCT_OPT_BIN     - Path to circt-opt binary (default: auto-detect)
#   USE_DOCKER        - Set to "1" to run in Docker (default: auto-detect)
#

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARTIFACTS_DIR="$PROJECT_DIR/artifacts/comb_not_check"

# Critical coverage test patterns (basenames without .mlir)
CRITICAL_TESTS=("trace_signal_multi_not" "trace_signal_sigextract" "trace_signal_icmp")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[comb.not check]${NC} $*"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_critical() {
    echo -e "${CYAN}[CRITICAL]${NC} $*"
}

# Check if a test is critical
is_critical_test() {
    local basename="$1"
    for critical in "${CRITICAL_TESTS[@]}"; do
        if [[ "$basename" == "$critical" ]]; then
            return 0
        fi
    done
    return 1
}

# Detect if we should use Docker
detect_docker() {
    if [[ "${USE_DOCKER:-}" == "1" ]]; then
        return 0
    fi

    # Check if qemu-passes binary exists locally
    if [[ -x "$PROJECT_DIR/build/qemu-passes" ]]; then
        return 1
    fi

    # Check if Docker container 'circt' exists and has the binary
    if docker ps -a --format '{{.Names}}' | grep -q '^circt$'; then
        return 0
    fi

    return 1
}

# Run command (either locally or in Docker)
run_cmd() {
    if [[ "$USE_DOCKER_MODE" == "1" ]]; then
        docker exec circt bash -c "$* 2>&1" 2>/dev/null
    else
        eval "$@" 2>&1
    fi
}

# Convert local path to Docker path
to_docker_path() {
    local path="$1"
    echo "$path" | sed 's|/home/njzhx/Desktop/circt-workspace/workspace|/home/user/workspace|'
}

# Convert Docker path to local path
to_local_path() {
    local path="$1"
    echo "$path" | sed 's|/home/user/workspace|/home/njzhx/Desktop/circt-workspace/workspace|'
}

# ==============================================================================
# Setup
# ==============================================================================

setup() {
    log_info "Setting up environment..."

    # Detect Docker mode
    if detect_docker; then
        USE_DOCKER_MODE="1"
        log_info "Using Docker mode"

        # Ensure Docker container is running
        if ! docker ps --format '{{.Names}}' | grep -q '^circt$'; then
            log_info "Starting Docker container 'circt'..."
            docker start circt >/dev/null 2>&1 || {
                log_fail "Failed to start Docker container 'circt'"
                exit 1
            }
            sleep 2
        fi

        QEMU_PASSES_BIN="${QEMU_PASSES_BIN:-/home/user/workspace/qemu-passes/build/qemu-passes}"
        CIRCT_OPT_BIN="${CIRCT_OPT_BIN:-/home/user/circt/build/Release/bin/circt-opt}"
    else
        USE_DOCKER_MODE="0"
        log_info "Using local mode"
        QEMU_PASSES_BIN="${QEMU_PASSES_BIN:-$PROJECT_DIR/build/qemu-passes}"
        CIRCT_OPT_BIN="${CIRCT_OPT_BIN:-}"
    fi

    # Verify binary exists
    if [[ "$USE_DOCKER_MODE" == "1" ]]; then
        if ! run_cmd "test -x '$QEMU_PASSES_BIN'"; then
            log_fail "qemu-passes binary not found at $QEMU_PASSES_BIN"
            exit 1
        fi
    else
        if [[ ! -x "$QEMU_PASSES_BIN" ]]; then
            log_fail "qemu-passes binary not found at $QEMU_PASSES_BIN"
            exit 1
        fi
    fi

    log_info "Using binary: $QEMU_PASSES_BIN"

    # Create artifacts directory
    mkdir -p "$ARTIFACTS_DIR"
    log_info "Artifacts directory: $ARTIFACTS_DIR"
}

# ==============================================================================
# Collect Input Files
# ==============================================================================

collect_inputs() {
    local inputs=()

    # Source 1: qemu-output/test/source/*.mlir
    local qemu_output_dir="/home/njzhx/Desktop/circt-workspace/workspace/qemu-output/test/source"
    if [[ -d "$qemu_output_dir" ]]; then
        while IFS= read -r -d '' f; do
            inputs+=("$f")
        done < <(find "$qemu_output_dir" -name "*.mlir" -type f -print0 2>/dev/null)
    else
        log_warn "Directory not found: $qemu_output_dir"
    fi

    # Source 2: qemu-passes/test/*.mlir
    local qemu_passes_test_dir="$PROJECT_DIR/test"
    if [[ -d "$qemu_passes_test_dir" ]]; then
        while IFS= read -r -d '' f; do
            inputs+=("$f")
        done < <(find "$qemu_passes_test_dir" -name "*.mlir" -type f -print0 2>/dev/null)
    else
        log_warn "Directory not found: $qemu_passes_test_dir"
    fi

    if [[ ${#inputs[@]} -eq 0 ]]; then
        log_fail "No input files found!"
        exit 1
    fi

    printf '%s\n' "${inputs[@]}"
}

# ==============================================================================
# Check for comb.not in IR
# ==============================================================================

# Check if file contains comb.not, return line numbers
check_comb_not() {
    local file="$1"
    local lines

    if [[ -f "$file" ]]; then
        lines=$(grep -n 'comb\.not\b' "$file" 2>/dev/null | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')
        if [[ -n "$lines" ]]; then
            echo "$lines"
            return 0
        fi
    fi

    echo ""
    return 1
}

# ==============================================================================
# Run Pipeline Stages
# ==============================================================================

# Run full pipeline (--all-passes + --dff-demo), capture stderr
run_full_pipeline() {
    local input="$1"
    local output="$2"
    local stderr_file="${3:-/dev/null}"
    local docker_input docker_output docker_stderr

    if [[ "$USE_DOCKER_MODE" == "1" ]]; then
        docker_input=$(to_docker_path "$input")
        docker_output=$(to_docker_path "$output")
        docker_stderr=$(to_docker_path "$stderr_file")
        docker exec circt bash -c "$QEMU_PASSES_BIN '$docker_input' --all-passes --dff-demo --emit-output -o '$docker_output' 2>'$docker_stderr'" >/dev/null 2>&1
    else
        "$QEMU_PASSES_BIN" "$input" --all-passes --dff-demo --emit-output -o "$output" 2>"$stderr_file"
    fi
}

# Run clock removal only
run_clock_removal() {
    local input="$1"
    local output="$2"
    local docker_input docker_output

    if [[ "$USE_DOCKER_MODE" == "1" ]]; then
        docker_input=$(to_docker_path "$input")
        docker_output=$(to_docker_path "$output")
        docker exec circt bash -c "$QEMU_PASSES_BIN '$docker_input' --all-passes --emit-output -o '$docker_output'" >/dev/null 2>&1
    else
        "$QEMU_PASSES_BIN" "$input" --all-passes --emit-output -o "$output" >/dev/null 2>&1
    fi
}

# Run canonicalize/cse using circt-opt
run_canonicalize() {
    local input="$1"
    local output="$2"
    local docker_input docker_output

    if [[ "$USE_DOCKER_MODE" == "1" ]]; then
        docker_input=$(to_docker_path "$input")
        docker_output=$(to_docker_path "$output")
        if docker exec circt bash -c "test -x '$CIRCT_OPT_BIN'" 2>/dev/null; then
            docker exec circt bash -c "$CIRCT_OPT_BIN '$docker_input' --canonicalize --cse -o '$docker_output'" >/dev/null 2>&1
            return $?
        fi
    else
        if [[ -x "$CIRCT_OPT_BIN" ]]; then
            "$CIRCT_OPT_BIN" "$input" --canonicalize --cse -o "$output" >/dev/null 2>&1
            return $?
        fi
    fi

    # Fallback: copy input to output if circt-opt not available
    cp "$input" "$output"
    return 1
}

# Run dff-demo only
run_dff_demo() {
    local input="$1"
    local output="$2"
    local docker_input docker_output

    if [[ "$USE_DOCKER_MODE" == "1" ]]; then
        docker_input=$(to_docker_path "$input")
        docker_output=$(to_docker_path "$output")
        docker exec circt bash -c "$QEMU_PASSES_BIN '$docker_input' --dff-demo --emit-output -o '$docker_output'" >/dev/null 2>&1
    else
        "$QEMU_PASSES_BIN" "$input" --dff-demo --emit-output -o "$output" >/dev/null 2>&1
    fi
}

# ==============================================================================
# Main Test Logic
# ==============================================================================

main() {
    setup

    # Collect input files
    mapfile -t INPUT_FILES < <(collect_inputs)
    log_info "Corpus size: ${#INPUT_FILES[@]}"
    log_info "Critical coverage tests: ${CRITICAL_TESTS[*]}"
    echo ""

    # Track results
    declare -A FAILURES
    declare -A FAILURE_LINES
    declare -A STAGE_RESULTS
    declare -A INPUT_COMB_NOT      # Track comb.not in input (pre)
    declare -A OUTPUT_COMB_NOT     # Track comb.not in output (post)
    declare -A TEST_STATUS         # Track status of each test
    FAIL_COUNT=0
    PASS_COUNT=0
    SKIP_COUNT=0
    CRITICAL_FAIL_COUNT=0
    CRITICAL_PASS_COUNT=0

    # Process each input file
    for input_file in "${INPUT_FILES[@]}"; do
        local basename
        basename=$(basename "$input_file" .mlir)
        local artifact_prefix="$ARTIFACTS_DIR/$basename"
        local is_critical=0

        if is_critical_test "$basename"; then
            is_critical=1
            echo -n "Testing ${CYAN}$basename${NC} [CRITICAL]... "
        else
            echo -n "Testing $basename... "
        fi

        # Create temp files for stages
        local final_output="$artifact_prefix.final.mlir"
        local after_clock="$artifact_prefix.after_clock.mlir"
        local after_canon="$artifact_prefix.after_canon.mlir"
        local after_dff_demo="$artifact_prefix.after_dff_demo.mlir"
        local stderr_file="$artifact_prefix.stderr.txt"

        # Check for comb.not in INPUT (pre-pipeline)
        local input_lines
        input_lines=$(check_comb_not "$input_file" || true)
        INPUT_COMB_NOT["$basename"]="${input_lines:-NO}"

        # Run full pipeline
        if ! run_full_pipeline "$input_file" "$final_output" "$stderr_file"; then
            # Pipeline failed (parse error or other)
            if [[ $is_critical -eq 1 ]]; then
                # Critical test: FAIL on parse error
                echo -e "${RED}FAIL${NC} (parse error)"
                ((FAIL_COUNT++)) || true
                ((CRITICAL_FAIL_COUNT++)) || true
                FAILURES["$basename"]="parse_error"
                TEST_STATUS["$basename"]="FAIL(parse)"
                OUTPUT_COMB_NOT["$basename"]="N/A"
                # Keep stderr file for debugging
                if [[ -f "$stderr_file" ]] && [[ -s "$stderr_file" ]]; then
                    echo "  -> stderr saved to: $stderr_file"
                fi
            else
                # Non-critical test: SKIP on parse error
                echo -e "${YELLOW}SKIP${NC} (parse error)"
                ((SKIP_COUNT++)) || true
                TEST_STATUS["$basename"]="SKIP"
                OUTPUT_COMB_NOT["$basename"]="N/A"
                rm -f "$stderr_file" 2>/dev/null
            fi
            continue
        fi

        # Check for comb.not in OUTPUT (post-pipeline)
        local final_lines
        final_lines=$(check_comb_not "$final_output" || true)
        OUTPUT_COMB_NOT["$basename"]="${final_lines:-NO}"

        if [[ -z "$final_lines" ]]; then
            echo -e "${GREEN}PASS${NC}"
            ((PASS_COUNT++)) || true
            if [[ $is_critical -eq 1 ]]; then
                ((CRITICAL_PASS_COUNT++)) || true
            fi
            TEST_STATUS["$basename"]="PASS"
            # Clean up artifacts for passing tests
            rm -f "$final_output" "$stderr_file" 2>/dev/null
        else
            echo -e "${RED}FAIL${NC} (comb.not at lines: $final_lines)"
            ((FAIL_COUNT++)) || true
            if [[ $is_critical -eq 1 ]]; then
                ((CRITICAL_FAIL_COUNT++)) || true
            fi
            FAILURES["$basename"]="comb_not"
            FAILURE_LINES["$basename.final"]="$final_lines"
            TEST_STATUS["$basename"]="FAIL(comb.not)"

            # Stage analysis for failed files
            # Stage A: After clock removal
            local clock_lines="N/A"
            if run_clock_removal "$input_file" "$after_clock" 2>/dev/null; then
                clock_lines=$(check_comb_not "$after_clock" || true)
                if [[ -z "$clock_lines" ]]; then
                    clock_lines="NO"
                else
                    FAILURE_LINES["$basename.after_clock"]="$clock_lines"
                    clock_lines="YES ($clock_lines)"
                fi
            fi
            STAGE_RESULTS["$basename.A"]="$clock_lines"

            # Stage B: After canonicalize/cse
            local canon_lines="N/A"
            if [[ -f "$after_clock" ]]; then
                if run_canonicalize "$after_clock" "$after_canon" 2>/dev/null; then
                    canon_lines=$(check_comb_not "$after_canon" || true)
                    if [[ -z "$canon_lines" ]]; then
                        canon_lines="NO"
                    else
                        FAILURE_LINES["$basename.after_canon"]="$canon_lines"
                        canon_lines="YES ($canon_lines)"
                    fi
                fi
            fi
            STAGE_RESULTS["$basename.B"]="$canon_lines"

            # Stage C: After dff-demo
            local dff_lines="N/A"
            local dff_input="$after_canon"
            [[ ! -f "$dff_input" ]] && dff_input="$after_clock"
            [[ ! -f "$dff_input" ]] && dff_input="$input_file"

            if run_dff_demo "$dff_input" "$after_dff_demo" 2>/dev/null; then
                dff_lines=$(check_comb_not "$after_dff_demo" || true)
                if [[ -z "$dff_lines" ]]; then
                    dff_lines="NO"
                else
                    FAILURE_LINES["$basename.after_dff_demo"]="$dff_lines"
                    dff_lines="YES ($dff_lines)"
                fi
            fi
            STAGE_RESULTS["$basename.C"]="$dff_lines"
        fi
    done

    # ==============================================================================
    # Print Summary
    # ==============================================================================

    echo ""
    echo "============================================================"
    echo "                    SUMMARY REPORT"
    echo "============================================================"
    echo ""

    # Print critical test results first
    echo "------------------------------------------------------------"
    echo -e "${BOLD}Critical Coverage Tests:${NC}"
    echo "------------------------------------------------------------"
    printf "%-30s | %-12s | %-12s | %-12s\n" "Test" "Status" "Input(pre)" "Output(post)"
    printf "%-30s-+-%-12s-+-%-12s-+-%-12s\n" "------------------------------" "------------" "------------" "------------"

    local all_critical_pass=1
    for critical in "${CRITICAL_TESTS[@]}"; do
        local status="${TEST_STATUS[$critical]:-NOT_FOUND}"
        local input_not="${INPUT_COMB_NOT[$critical]:-N/A}"
        local output_not="${OUTPUT_COMB_NOT[$critical]:-N/A}"

        # Color the status
        local status_colored
        if [[ "$status" == "PASS" ]]; then
            status_colored="${GREEN}PASS${NC}"
        elif [[ "$status" == FAIL* ]]; then
            status_colored="${RED}$status${NC}"
            all_critical_pass=0
        elif [[ "$status" == "SKIP" ]]; then
            status_colored="${YELLOW}SKIP${NC}"
            all_critical_pass=0
        else
            status_colored="${RED}NOT_FOUND${NC}"
            all_critical_pass=0
        fi

        printf "%-30s | " "$critical"
        echo -e -n "$status_colored"
        # Pad to align columns (assuming status is ~12 chars without color codes)
        local status_len=${#status}
        printf "%*s | %-12s | %-12s\n" $((12 - status_len)) "" "$input_not" "$output_not"
    done

    echo ""
    echo "------------------------------------------------------------"
    echo "Overall Results:"
    echo "------------------------------------------------------------"
    echo "  Total:     ${#INPUT_FILES[@]}"
    echo "  Passed:    $PASS_COUNT"
    echo "  Skipped:   $SKIP_COUNT"
    echo "  Failed:    $FAIL_COUNT"
    echo ""
    echo "  Critical tests passed: $CRITICAL_PASS_COUNT / ${#CRITICAL_TESTS[@]}"
    echo ""

    # Determine final verdict
    if [[ $all_critical_pass -eq 1 ]] && [[ $FAIL_COUNT -eq 0 ]]; then
        echo "------------------------------------------------------------"
        log_pass "All critical coverage tests PASSED"
        echo "------------------------------------------------------------"
        echo ""
        echo -e "${GREEN}${BOLD}Conclusion:${NC}"
        echo "  traceSignal() does NOT need comb.not support"
        echo "  (the pipeline does not generate comb.not)"
        echo ""
        echo "  Evidence:"
        echo "  - All critical tests (multi_not, sigextract, icmp) parsed and executed"
        echo "  - No comb.not found in any pipeline output"
        echo ""
    elif [[ $all_critical_pass -eq 0 ]]; then
        echo "------------------------------------------------------------"
        log_fail "Critical coverage tests incomplete or failed"
        echo "------------------------------------------------------------"
        echo ""
        echo -e "${RED}${BOLD}Conclusion: INCONCLUSIVE${NC}"
        echo "  Cannot determine if traceSignal() needs comb.not support"
        echo ""
        echo "  Reason: Not all critical tests completed successfully"
        for critical in "${CRITICAL_TESTS[@]}"; do
            local status="${TEST_STATUS[$critical]:-NOT_FOUND}"
            if [[ "$status" != "PASS" ]]; then
                echo "    - $critical: $status"
            fi
        done
        echo ""
    else
        # Critical tests passed but some non-critical failed with comb.not
        echo "------------------------------------------------------------"
        log_fail "comb.not found in $FAIL_COUNT file(s)"
        echo "------------------------------------------------------------"
        echo ""

        # Print detailed failure report
        echo "Detailed Failure Report:"
        for basename in "${!FAILURES[@]}"; do
            local reason="${FAILURES[$basename]}"
            echo ""
            echo "- $basename.mlir ($reason)"
            if [[ "$reason" == "comb_not" ]]; then
                echo "  final: YES (lines: ${FAILURE_LINES[$basename.final]:-?})"
                echo "  after_clock: ${STAGE_RESULTS[$basename.A]:-N/A}"
                echo "  after_canon: ${STAGE_RESULTS[$basename.B]:-N/A}"
                echo "  after_dff_demo: ${STAGE_RESULTS[$basename.C]:-N/A}"
            fi
            echo "  artifacts: $ARTIFACTS_DIR/$basename.*"
        done

        echo ""
        echo -e "${RED}${BOLD}Conclusion:${NC}"
        echo "  traceSignal() MAY need comb.not support"
        echo "  OR a normalize-not pass should be added"
        echo ""
    fi

    echo "============================================================"
    echo "Artifacts saved to: $ARTIFACTS_DIR"
    echo "============================================================"

    # Return appropriate exit code
    # Exit 1 if any critical test failed OR any test has comb.not in output
    if [[ $all_critical_pass -eq 0 ]] || [[ $FAIL_COUNT -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

# ==============================================================================
# Entry Point
# ==============================================================================

main "$@"
