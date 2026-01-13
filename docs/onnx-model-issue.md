# ONNX Model Export Issue & Solution

## Problem Summary

The current ONNX model (`model_int8.onnx`) has a **fundamental architectural issue** that prevents it from generating proper captions in the browser.

### What's Wrong

**Current ONNX Model:**
- Input: `pixel_values` [1, 3, 224, 224]
- Output: `logits` [1, **196**, 49152]

The model outputs logits for **196 vision patch positions**, not for generated text tokens!

**Test Result:**
```python
# Running inference produces:
Token IDs: [28, 28, 198, 28, 96, 28, 2382, 429, ...]
Decoded: ",,\n,p, red from,ce..."  # Gibberish!
```

### Root Cause

The ONNX export (`export_onnx_quantized.py` lines 109-124) only captures **ONE forward pass**:

```python
class ONNXExportWrapper(torch.nn.Module):
    def forward(self, pixel_values):
        outputs = self.model(pixel_values)  # Single pass
        return outputs.logits  # Logits for vision tokens!
```

This is **NOT** the same as the Python `.generate()` method which does:
1. Encode image → vision embeddings (196 tokens)
2. Use vision embeddings as **prefix**
3. Generate text tokens **autoregressively** (one at a time)
4. Stop at EOS token

## Why Python Works but ONNX Doesn't

**Python (transformers pipeline):**
```python
pipeline("image-to-text", model="tarekziade/distilvit2")(image)
# Uses model.generate() → proper autoregressive generation
# Output: "a red square on a white background"
```

**ONNX (current):**
```javascript
session.run({pixel_values})
// Single forward pass → logits for vision positions only
// Output: gibberish
```

## Solutions

### Option 1: Re-export with Separate Components (Recommended)

Export three separate ONNX models:

1. **Vision Encoder**: `pixel_values` → `vision_features` [1, 196, 768]
2. **Projection**: `vision_features` → `vision_embeds` [1, 196, 576]
3. **Decoder** (autoregressive):
   - Input: `inputs_embeds` [1, seq_len, 576], `past_key_values`
   - Output: `logits` [1, 1, 49152], `past_key_values`

Then implement generation loop in JavaScript:

```javascript
// 1. Encode image
const visionFeatures = await visionEncoder.run({pixel_values});

// 2. Project to text space
const visionEmbeds = await projection.run({vision_features: visionFeatures});

// 3. Generate tokens one by one
let generated = [];
let past_kv = null;

for (let i = 0; i < maxLength; i++) {
    const inputs = i === 0 ? visionEmbeds : getEmbedding(generated[i-1]);
    const outputs = await decoder.run({inputs_embeds: inputs, past_key_values: past_kv});

    const nextToken = argmax(outputs.logits[0][-1]);
    generated.push(nextToken);

    past_kv = outputs.past_key_values;

    if (nextToken === EOS_TOKEN) break;
}
```

### Option 2: Export with Generation Loop Baked In (Complex)

Requires exporting the entire generation loop as part of the ONNX graph. This is very difficult because:
- ONNX doesn't support Python-level loops well
- Requires dynamic control flow (if statements for EOS detection)
- Much larger model size
- Harder to optimize

### Option 3: Use the Python Model Directly

Skip ONNX entirely and use:
- **transformers.js native support** (if/when PrefixConditioningVLM is added)
- **ONNX Runtime with Python backend** (defeats the purpose of browser deployment)
- **Server-side API** with Python model

## Recommended Fix

**Step 1:** Re-export the model properly

Use the consolidated `export_prefix_vlm.py` script which exports the four components needed for JS:

```bash
python export_prefix_vlm.py --model-dir ./siglip-base-patch16-224-SmolLM-135M-lora --output-dir onnx
```

This creates:
```
onnx/
├── vision_encoder/model.onnx     # Vision encoder only
├── projection.onnx               # Projection layer only
├── language_model/model.onnx     # Decoder with past_key_values
└── prefix_init.onnx              # First pass (prefix embeddings + prompt ids → logits + cache)
```

**Step 2:** Implement JavaScript generation loop

Update `demo.html` to:
1. Load all three models
2. Run vision encoder + projection once
3. Implement autoregressive loop with decoder

**Step 3:** Handle KV caching

For efficiency, the decoder should support past_key_values caching to avoid recomputing attention for previous tokens.

## Quick Fix for Current Demo

The current demo.html won't work properly until the ONNX model is re-exported. As a temporary measure, add a warning:

```javascript
// In demo.html
showError(
    'Model export issue detected. ' +
    'The current ONNX model cannot generate captions properly. ' +
    'See docs/onnx-model-issue.md for details.'
);
```

## Timeline Estimate

- **Re-export with separate components**: 2-4 hours
- **Implement JavaScript generation**: 4-8 hours
- **Test and debug**: 2-4 hours
- **Total**: 1-2 days of focused work

## References

- [docs/onnx-export-guide.md](onnx-export-guide.md) - Original export documentation (legacy)
- [export_prefix_vlm.py](../export_prefix_vlm.py) - Current export script
