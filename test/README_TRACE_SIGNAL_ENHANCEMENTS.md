# traceSignal() Enhancement Documentation

## Overview

The `traceSignal()` function in `DffDemoPass` has been enhanced to support more complex signal tracing patterns while maintaining backward compatibility.

## Enhancements

### 1. Multi-Layer NOT Support (Odd/Even Parity)

**Feature**: Traces through multiple layers of NOT operations and computes inversion parity.

**Supported NOT Forms**:
- `comb.xor %x, %allones` - XOR with all-ones constant
- `comb.xor %allones, %x` - XOR with constant on left
- Chains of the above (e.g., `xor(xor(x, 1), 1)`)

**Behavior**:
- Odd number of NOTs → signal inverted (isInverted=true)
- Even number of NOTs → signal not inverted (isInverted=false)
- Max depth: 32 layers (prevents infinite loops)

**Example**:
```mlir
%prb = llhd.prb %signal : !hw.inout<i1>
%inv1 = comb.xor %prb, %c1_i1 : i1      // First NOT
%inv2 = comb.xor %inv1, %c1_i1 : i1     // Second NOT (cancels)

// traceSignal(%inv2) returns ("signal", false) - even count = not inverted
```

**Test Case**: `test/test_double_xor.mlir`

### 2. ICmp-Based NOT Patterns (i1 Semantics)

**Feature**: Recognizes comparison operations that implement NOT for i1 types.

**Supported Patterns**:
| Pattern | Equivalent | Inversion |
|---------|-----------|-----------|
| `comb.icmp eq %x, 0` | `!x` | Yes |
| `comb.icmp ne %x, 0` | `x` | No |
| `comb.icmp eq %x, 1` | `x` | No |
| `comb.icmp ne %x, 1` | `!x` | Yes |

**Requirements**:
- Only works for i1 (1-bit) comparisons
- Constant must be 0 or 1
- Works with constant on either side

**Example**:
```mlir
%prb = llhd.prb %signal : !hw.inout<i1>
%cmp = comb.icmp eq %prb, %c0_i1 : i1  // x == 0 means !x

// traceSignal(%cmp) returns ("signal", true) - inverted
```

**Limitation**: Non-i1 or non-0/1 comparisons stop tracing (return empty name).

### 3. SigExtract Chain Unwrapping

**Feature**: Penetrates through `llhd.sig.extract` chains to find base signal.

**Before**:
```cpp
if (defOp->getName().getStringRef() == "llhd.sig") {  // String comparison
  ...
}
```

**After**:
```cpp
sig = unwrapSigExtract(sig);  // Unwrap chains
if (auto sigOp = sig.getDefiningOp<llhd::SignalOp>()) {  // Type-based check
  ...
}
```

**Benefit**: Correctly traces signals extracted from wider buses.

**Example**:
```mlir
%bus = llhd.sig name "ctrl_bus" %c0_i8 : i8
%bit0 = llhd.sig.extract %bus from 0 : !hw.inout<i8> -> !hw.inout<i1>
%prb = llhd.prb %bit0 : !hw.inout<i1>

// traceSignal(%prb) now returns ("ctrl_bus", false) - correctly unwrapped
```

### 4. BlockArgument (Module Port) Support

**Feature**: Attempts to extract signal names from module ports (BlockArguments).

**Supported**:
- `hw.module` ports via `getArgName(idx)`

**Not Supported** (returns empty name):
- Other module types without name API

**Example**:
```mlir
hw.module @test(%pwrite_port: !hw.inout<i1>) {
  %prb = llhd.prb %pwrite_port : !hw.inout<i1>
  // traceSignal(%prb) returns ("pwrite_port", false)
}
```

## Implementation Details

### Algorithm

```cpp
std::pair<StringRef, bool> traceSignal(Value val) {
  bool invertParity = false;
  Value cur = val;

  // Phase 1: Peel NOT layers (max depth 32)
  for (int depth = 0; depth < 32; ++depth) {
    if (comb::XorOp xor with allones constant) {
      invertParity = !invertParity;
      cur = non-constant operand;
      continue;
    }
    if (comb::ICmpOp with 0/1 constant on i1) {
      Update invertParity based on predicate;
      cur = non-constant operand;
      continue;
    }
    break;  // No more layers
  }

  // Phase 2: Extract signal name
  if (llhd::PrbOp prb = cur.getDefiningOp()) {
    Value sig = unwrapSigExtract(prb.getSignal());
    if (llhd::SignalOp sigOp = sig.getDefiningOp()) {
      return {sigOp.getName(), invertParity};
    }
    if (BlockArgument port) {
      return {getModulePortName(port), invertParity};
    }
  }

  return {StringRef(), false};  // Not traceable
}
```

### Limitations

1. **Seen Map Limitation**: The caller's `seen` map (in `DffDemoPass::runOnOperation`) only stores one boolean per signal name. Complex inversion patterns like:
   ```
   %a = and(!signal, signal)  // Should be always false
   ```
   are **not** handled - only the first occurrence is recorded.

2. **Max Depth**: 32-layer NOT depth limit prevents infinite loops but may miss pathological cases.

3. **ICmp Scope**: Only handles simple i1 comparisons with 0/1 constants. Complex predicates (slt, ult, etc.) are not folded.

## Testing

### Test Files

1. **test/test_double_xor.mlir**
   - Tests: Double XOR cancellation (even parity)
   - Expected: `pwrite=1, psel=1` (no inversion)

2. **test/trace_signal_multi_not.mlir** (syntax example)
   - Shows multi-layer NOT scenarios
   - Note: May need syntax fixes for actual execution

3. **test/trace_signal_icmp.mlir** (syntax example)
   - Demonstrates icmp-based NOT patterns

4. **test/trace_signal_sigextract.mlir** (syntax example)
   - Shows SigExtract unwrapping

### Running Tests

```bash
docker exec circt bash -c "/home/user/workspace/qemu-passes/build/qemu-passes \
  test/test_double_xor.mlir --dff-demo"
```

**Expected Output**:
```
[Found Trigger Condition] at ...
  Path type: Write (pwrite=1)
  To activate this path, set:
    pwrite = 1    # Double inversion canceled
    psel = 1
```

## Backward Compatibility

- **Preserved**: All existing pass logic (AND detection, pwrite requirement, seen deduplication)
- **Enhanced**: Signal tracing is now more robust without breaking existing behavior
- **Tested**: gpio0_simplified.mlir produces identical results (18 signal replacements)

## Future Work

1. **Seen Map Enhancement**: Track both inverted and non-inverted uses per signal
2. **Wider ICmp Support**: Handle more predicates and bit widths
3. **LLHD Entity Support**: Add name extraction for llhd.entity ports
4. **Caching**: Memoize traceSignal results for performance

## References

- **Modified File**: `lib/DffDemoPass.cpp`
- **Helper Function**: `unwrapSigExtract(Value sig)`
- **Main Function**: `traceSignal(Value val)`
- **Pass Definition**: `include/Passes.td` (DffDemo pass)
