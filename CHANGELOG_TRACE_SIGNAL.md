# traceSignal() Enhancement Changelog

## Summary

Enhanced `traceSignal()` function in `lib/DffDemoPass.cpp` to support:
1. **Multi-layer NOT** (comb.xor chains with odd/even parity tracking)
2. **ICmp-based NOT** (i1 comparisons with 0/1 constants)
3. **SigExtract unwrapping** (penetrate llhd.sig.extract chains)
4. **Type-based signal detection** (use llhd::SignalOp instead of string comparison)
5. **BlockArgument support** (extract names from hw.module ports)

## Files Modified

### lib/DffDemoPass.cpp
**Added**:
- `unwrapSigExtract(Value sig)` - Helper to unwrap llhd.sig.extract chains
- Enhanced `traceSignal(Value val)` - Multi-layer NOT support with max depth 32

**Changes**:
- Removed: ~~`comb::NotOp`~~ (doesn't exist in Comb dialect)
- Added: XOR with constant on left or right
- Added: ICmp eq/ne with 0/1 for i1 types
- Added: Loop-based NOT peeling (max 32 layers)
- Changed: Signal name extraction uses `llhd::SignalOp` type check
- Changed: Unwraps `llhd::SigExtractOp` chains before extracting name
- Added: BlockArgument name extraction for hw.module ports

### test/test_double_xor.mlir (NEW)
- Tests double XOR cancellation
- Verifies even parity returns non-inverted signal
- **Status**: ✓ Passes (detects pwrite=1, psel=1)

### test/README_TRACE_SIGNAL_ENHANCEMENTS.md (NEW)
- Comprehensive documentation of enhancements
- Usage examples and test cases
- Limitations and future work

## Build & Test

```bash
# Build
docker exec circt bash -c "cd /home/user/workspace/qemu-passes/build && \
  cmake .. -DCMAKE_PREFIX_PATH='/home/user/circt/llvm/build/Release;/home/user/circt/build/Release' && \
  make -j\$(nproc)"

# Test double XOR
docker exec circt bash -c "/home/user/workspace/qemu-passes/build/qemu-passes \
  test/test_double_xor.mlir --dff-demo"

# Test on gpio0 (backward compatibility)
docker exec circt bash -c "/home/user/workspace/qemu-passes/build/qemu-passes \
  results/gpio0_simplified.mlir --dff-demo"
```

## Verification Results

### Test: test_double_xor.mlir
```
[Found Trigger Condition] at test_double_xor.mlir:20:14
  Path type: Write (pwrite=1)
  To activate this path, set:
    pwrite = 1        # ✓ Double inversion canceled
    psel = 1
```
✅ **PASS** - Correctly detected pwrite=1 (even parity)

### Test: gpio0_simplified.mlir
```
[Optimization] ✓ Replaced 18 signal(s) with constants
[DCE] ✓ Removed 18 dead probe(s), 1 dead XOR(s)
```
✅ **PASS** - Same results as before (backward compatible)

## API Changes

### Before
```cpp
std::pair<StringRef, bool> traceSignal(Value val) {
  bool isInverted = false;

  // Only handles single XOR with constant on RHS
  if (auto xorOp = val.getDefiningOp<comb::XorOp>()) {
    if (rhs is allones constant) {
      isInverted = true;
      val = lhs;
    }
  }

  // Only handles llhd.sig (string check)
  if (auto prbOp = val.getDefiningOp<llhd::PrbOp>()) {
    if (defOp->getName() == "llhd.sig") {
      return {name, isInverted};
    }
  }

  return {StringRef(), false};
}
```

### After
```cpp
std::pair<StringRef, bool> traceSignal(Value val) {
  bool invertParity = false;
  Value cur = val;

  // Peel multiple NOT layers (max 32)
  for (int depth = 0; depth < 32; ++depth) {
    if (comb::XorOp with allones - either side) {
      invertParity = !invertParity;
      cur = other operand;
      continue;
    }

    if (comb::ICmpOp eq/ne with 0/1 on i1) {
      update invertParity;
      cur = other operand;
      continue;
    }

    break;  // No more layers
  }

  // Extract name with SigExtract unwrapping
  if (auto prbOp = cur.getDefiningOp<llhd::PrbOp>()) {
    Value sig = unwrapSigExtract(prb.getSignal());

    if (auto sigOp = sig.getDefiningOp<llhd::SignalOp>()) {  // Type check
      return {sigOp.getName(), invertParity};
    }

    if (auto port = dyn_cast<BlockArgument>(sig)) {
      return {getPortName(port), invertParity};  // Module port
    }
  }

  return {StringRef(), false};
}
```

## Key Improvements

| Feature | Before | After |
|---------|--------|-------|
| XOR constant side | RHS only | LHS or RHS |
| NOT layers | 1 | Up to 32 (with parity) |
| ICmp support | ❌ | ✅ (i1 with 0/1) |
| SigExtract | ❌ | ✅ (unwraps chains) |
| Signal detection | String "llhd.sig" | Type `llhd::SignalOp` |
| Module ports | ❌ | ✅ (hw.module only) |

## Backward Compatibility

✅ **Fully compatible** - No changes to:
- DffDemoPass main logic
- AND detection with pwrite requirement
- `seen` map deduplication strategy
- Optimization and DCE phases

## Limitations

1. **Seen map**: Only stores one bool per signal name (doesn't track complex mixed inversion)
2. **Max depth**: 32-layer limit (prevents infinite loops)
3. **ICmp scope**: Only i1 with 0/1 constants
4. **Port support**: Only hw.module (not llhd.entity or other types)

## Next Steps

- ✅ Code compiled and tested
- ✅ Backward compatibility verified
- ✅ Documentation written
- 🔲 Additional test cases (icmp, sigextract) - syntax needs refinement
- 🔲 Consider seen map enhancement for complex cases
- 🔲 Add llhd.entity port support if needed

---

**Author**: Claude Sonnet 4.5
**Date**: 2026-01-09
**Commit**: Ready for `git add` and commit
