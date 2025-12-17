# Pruning Diagnostics: Verify "Both Pruned" Mode

## Your Concern is Valid! 

You're absolutely right to question whether `prune_training_model=True` actually prunes the training model. Let me explain the flow and how to verify it.

---

## The Flow (ASYNC Mode)

### Timeline of Events

```
Epoch 30:
├─ [START] Training happens on model (state unknown)
├─ [TRIGGER] maybe_trigger(30) called
│   └─ Async compression thread starts in background
├─ Training continues...
└─ [END] Epoch 30 completes

Async Thread (running in parallel):
├─ snapshot() extracts weights from training model
│   └─ 📸 Prints: "Snapshot taken - Training model is FULL/PRUNED"
├─ compress() creates pruned model
└─ Marks ready for swap

Epoch 31:
├─ [START] Training still on old model
├─ [SWAP] maybe_swap() detects compression complete
│   ├─ Broadcasts pruned model to inference workers ✅
│   ├─ Calls _is_structure_changing_compression()
│   │   └─ If prune_training_model=True:
│   │       └─ Calls _apply_masks_to_training()
│   │           ├─ Applies masks to local training model
│   │           ├─ Prints: "✅ Masks applied to training model"
│   │           └─ Prints: "📊 Training model sparsity: XX.X%"
│   └─ Now training model is PRUNED! 🎉
├─ Training continues on PRUNED model
└─ [END] Epoch 31 completes

Epoch 32:
├─ snapshot() will now see PRUNED training model
│   └─ 📸 Prints: "Snapshot taken - Training model sparsity: XX.X%"
└─ This confirms pruning persisted!
```

---

## How to Verify It's Working

### 1. Look for These Log Messages

When you run with `prune_training_model=True`, you should see:

```
Epoch 30:
[MaskPrune] 📸 Snapshot taken - Training model is FULL (not pruned)
...compression happens in background...

Epoch 31:
[PolicyManager] 🔥 Both Pruned mode:
                🎯 Training: Pruned model (sparsity: 40.0%)
                🎯 Inference: Pruned model (sparsity: 40.0%)
[PolicyManager] ✅ Masks applied to training model (Both Pruned mode)
[PolicyManager] 📊 Training model sparsity verified: 40.0%

Epoch 32:
[MaskPrune] 📸 Snapshot taken - Training model sparsity: 40.0%  ← KEY!
```

**Critical Check:** If you see "Snapshot taken - Training model sparsity: 40.0%" in epoch 32+, then pruning is working!

---

### 2. Check Performance Summary

At the end of training, look for:

```
=== Summary (async) ===
Epochs: 100
Reward mean (avg over epochs): 450.23
Total time (avg per epoch): 0.21s
  Rollout time avg: 0.19s | Train time avg: 0.02s
Throughput (avg samples/s): 9500.5
Final sparsity: 40.0%  ← Should match your prune_ratio!
```

---

### 3. Check Individual Epoch Logs

In your log file (`logs/pruning_basic/async_prune_compile/async_XXXXX.jsonl`), look for sparsity field:

```json
{"epoch": 32, "sparsity": 0.4, ...}
{"epoch": 33, "sparsity": 0.4, ...}
```

If `"sparsity": null`, something went wrong!

---

## What If It's NOT Working?

### Problem: Snapshot always says "Training model is FULL"

**Diagnosis:**
- `_apply_masks_to_training()` is not being called
- OR masks are not persisting (gradient hooks missing)

**Fix:**
Check logs for:
```
[PolicyManager] 🔥 Both Pruned mode:  ← Should appear!
[PolicyManager] ✅ Masks applied...   ← Should appear!
```

If missing, then `prune_training_model=True` is not being passed correctly.

---

### Problem: Snapshot shows sparsity once, then goes back to 0%

**Diagnosis:**
- Masks are applied but gradient hooks are not working
- Optimizer is restoring masked weights to non-zero

**Fix:**
This should NOT happen because `apply_masks_to_backbone(..., inference_only=False)` registers gradient hooks that prevent optimizer from updating masked weights.

---

## Architecture Comparison

### Mode 1: Teacher-Student (prune_training_model=False)

```
Snapshot Flow:
training_model (FULL) → snapshot (FULL) → compress → pruned_model (PRUNED)
                ↑                                            ↓
                └────────────────────────────────────────────┘
              (training model never changes)        (only inference uses pruned)
```

**Snapshot will ALWAYS show:** "Training model is FULL"

---

### Mode 2: Both Pruned (prune_training_model=True)

```
First Compression (Epoch 31):
training_model (FULL) → snapshot (FULL) → compress → pruned_model (PRUNED)
                ↓                                            ↓
        apply_masks()                                (inference uses pruned)
                ↓
training_model (PRUNED)

Next Compression (Epoch 46):
training_model (PRUNED) → snapshot (PRUNED) → compress → pruned_model (PRUNED)
        ↑                                                        ↓
        └────────────────────────────────────────────────────────┘
    (training model stays pruned)                   (inference uses pruned)
```

**Snapshot will show after first compression:** "Training model sparsity: 40.0%"

---

## Quick Test Script

Run this to verify pruning is working:

```bash
# Make sure prune_training_model=True in config
grep "prune_training_model" config_pruning.py

# Run experiment
python scripts/run_pruning_experiments.py --experiment basic

# Check logs for the diagnostic messages
grep "📸 Snapshot" logs/pruning_basic/async_prune_compile/*.jsonl
grep "🔥 Both Pruned" logs/pruning_basic/async_prune_compile/*.jsonl
grep "✅ Masks applied" logs/pruning_basic/async_prune_compile/*.jsonl
```

---

## Expected Output (if working correctly)

```bash
$ grep "📸 Snapshot" logs/.../async_prune_compile/*.jsonl

Epoch 30: [MaskPrune] 📸 Snapshot taken - Training model is FULL (not pruned)
Epoch 31: (no snapshot - swap happening)
Epoch 32: [MaskPrune] 📸 Snapshot taken - Training model sparsity: 40.0%  ✅
Epoch 45: [MaskPrune] 📸 Snapshot taken - Training model sparsity: 40.0%  ✅
Epoch 46: (compression triggered again)
Epoch 47: [MaskPrune] 📸 Snapshot taken - Training model sparsity: 40.0%  ✅
```

**If you see sparsity increasing from 0% → 40%, then it's working!** 🎉

---

## Key Insight

The "lag" you identified is real:
- **Epoch 30:** Trigger compression (snapshot shows FULL)
- **Epoch 31:** Swap & apply masks (training model becomes PRUNED)
- **Epoch 32+:** Training continues on PRUNED model (snapshot shows PRUNED)

This 1-epoch lag is **expected and correct** in ASYNC mode. The important thing is that **once masks are applied, they persist** via gradient hooks!

---

## If You Still Don't Trust It...

Add this to verify gradient hooks are working:

```python
# In framework/policy_manager.py, after line 642
apply_masks_to_backbone(training_backbone, masks, inference_only=False)

# Add diagnostic:
for i, layer in enumerate(training_backbone.hidden_layers):
    has_hook = hasattr(layer.weight, '_mask_hook_handle')
    print(f"  Layer {i}: gradient hook = {'✅ YES' if has_hook else '❌ NO'}")
```

Expected output:
```
Layer 0: gradient hook = ✅ YES
Layer 1: gradient hook = ✅ YES
Layer 2: gradient hook = ✅ YES
Layer 3: gradient hook = ✅ YES
```

---

## Summary

✅ **Added Diagnostics:**
1. Snapshot now prints training model sparsity
2. _apply_masks_to_training() now verifies sparsity after application  
3. Performance summary now includes final sparsity
4. All epoch logs now include sparsity field

✅ **How to Verify:**
Look for snapshot messages showing training model sparsity > 0% after first compression completes

✅ **The Flow IS Correct:**
- Masks ARE applied to training model when `prune_training_model=True`
- Gradient hooks DO prevent optimizer from breaking sparsity
- There IS a 1-epoch lag in ASYNC mode (this is expected)

**Run your experiment and check the logs - the diagnostics will tell you if it's working!** 🔍

