# Transformers.js Implementation of Prefix Conditioning VLM

This document describes the JavaScript/Transformers.js implementation of the prefix-conditioning vision-language model.

## ✨ Live Demo Available!

A complete working browser demo is available at **`demo.html`** in the project root.

**Quick start:**
```bash
python -m http.server 8000
open http://localhost:8000/demo.html
```

See the browser demo in `/Users/tarekziade/Dev/huggingface/distilvit2/demo` (main.js) for a working reference.

## Overview

The `distilvit/prefix_model.js` file provides a JavaScript implementation that mirrors the Python/PyTorch architecture, designed for browser deployment using ONNX Runtime.

## Architecture Mapping

### Python (PyTorch) → JavaScript (ONNX)

| Component | Python | JavaScript | ONNX Export |
|-----------|--------|------------|-------------|
| Vision Encoder | `SigLIP2Model` (frozen) | `vision_encoder` session | `vision_encoder.onnx` |
| Projection | `ProjectionHead` (Linear/MLP) | `projection` session | `projection.onnx` |
| Language Model | `SmolLM + LoRA` | `decoder` session | `decoder_model_merged.onnx` |

### Key Differences

1. **LoRA Merging**: In Python, LoRA adapters are applied dynamically. For ONNX export, LoRA weights must be merged into the base model first.

2. **Embedding Layer**: The JavaScript version assumes embeddings are pre-computed or handled separately, as ONNX export may separate the embedding layer.

3. **Generation**: The Python version uses HuggingFace's `.generate()` with full beam search. The JavaScript version implements a simplified autoregressive loop (requires completion).

## File Structure

```
distilvit2/
├── distilvit/
│   ├── prefix_model.py          # Python implementation
│   └── prefix_model.js          # JavaScript implementation (NEW)
└── docs/
    └── transformers-js-implementation.md  # This file
```

## Usage Example

### Basic Usage

```javascript
import { PrefixConditioningVLM } from './distilvit/prefix_model.js';
import { AutoTokenizer, AutoProcessor, RawImage } from '@huggingface/transformers';

// Load model and tokenizer
const model = await PrefixConditioningVLM.from_pretrained('path/to/model');
const tokenizer = await AutoTokenizer.from_pretrained('path/to/model');
const processor = await AutoProcessor.from_pretrained('path/to/model');

// Set tokenizer on model
model.tokenizer = tokenizer;

// Prepare image
const image = await RawImage.read('image.jpg');
const image_inputs = await processor(image);

// Generate caption
const output_ids = await model.generate({
    pixel_values: image_inputs.pixel_values,
    max_new_tokens: 30,
    num_beams: 3,
});

// Decode caption
const caption = tokenizer.decode(output_ids[0], { skip_special_tokens: true });
console.log('Caption:', caption);
```

### Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
    <title>Image Captioning with Prefix Conditioning VLM</title>
</head>
<body>
    <input type="file" id="imageInput" accept="image/*">
    <div id="output"></div>

    <script type="module">
        import { PrefixConditioningVLM } from './distilvit/prefix_model.js';
        import { AutoTokenizer, AutoProcessor, RawImage } from '@huggingface/transformers';

        // Load model once
        const model = await PrefixConditioningVLM.from_pretrained('model-name');
        const tokenizer = await AutoTokenizer.from_pretrained('model-name');
        const processor = await AutoProcessor.from_pretrained('model-name');
        model.tokenizer = tokenizer;

        document.getElementById('imageInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            const image = await RawImage.fromBlob(file);
            const inputs = await processor(image);

            const output_ids = await model.generate({
                pixel_values: inputs.pixel_values,
                max_new_tokens: 30,
            });

            const caption = tokenizer.decode(output_ids[0], { skip_special_tokens: true });
            document.getElementById('output').textContent = caption;
        });
    </script>
</body>
</html>
```

## ONNX Export Requirements

To use the JavaScript implementation, you need to export the Python model to ONNX format.

### Export Script Template

```python
import torch
from transformers import AutoModel
from distilvit.prefix_model import PrefixConditioningVLM

# Load trained model
model = PrefixConditioningVLM.from_pretrained('path/to/checkpoint')
model.eval()

# 1. Export vision encoder
dummy_image = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model.vision_encoder,
    dummy_image,
    "vision_encoder.onnx",
    input_names=['pixel_values'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'pixel_values': {0: 'batch_size'},
        'last_hidden_state': {0: 'batch_size'},
    }
)

# 2. Export projection layer
dummy_vision_features = torch.randn(1, 729, 1152)  # [B, num_patches, vision_dim]
torch.onnx.export(
    model.projection,
    dummy_vision_features,
    "projection.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
)

# 3. Merge LoRA and export language model
from peft import merge_and_unload
language_model = model.language_model.merge_and_unload()

# Export decoder (this is complex - see docs/onnx-export-guide.md)
# You'll need to export with past_key_values support for efficient generation
```

**Important**: Exporting decoder models with KV cache is non-trivial. See:
- [docs/onnx-export-guide.md](onnx-export-guide.md) for detailed instructions
- [Optimum](https://github.com/huggingface/optimum) for automated ONNX export

## Implementation Status

### ✅ Completed

- [x] Model architecture (PrefixConditioningVLM class)
- [x] Configuration handling (PrefixConditioningConfig)
- [x] Vision encoding (encode_image)
- [x] Projection layer integration (project_vision_features)
- [x] Forward pass structure
- [x] Generation method signature

### ⚠️ Requires Integration

These components are **placeholders** that need integration with transformers.js utilities:

- [ ] **Tensor operations**: Replace placeholder methods with actual transformers.js tensor ops
  - `_concatenate_tensors` → use `cat()` from transformers.js
  - `_ones` → use `ones()` from transformers.js
  - Tensor manipulation utilities

- [ ] **ONNX session loading**: Implement `from_pretrained` using transformers.js hub utilities
  - Load ONNX files from HuggingFace Hub or local path
  - Initialize inference sessions
  - Handle device/dtype options

- [ ] **Generation loop**: Complete the autoregressive generation
  - Beam search implementation
  - KV cache management
  - Logit processors (temperature, top-p, top-k)
  - Sampling methods

- [ ] **Embedding layer**: Handle token → embedding conversion
  - May need separate ONNX export of embedding layer
  - Or use pre-computed embeddings approach

## Integration with Transformers.js

To fully integrate this with transformers.js, you would:

1. **Add to models.js**: Register the model class in transformers.js

```javascript
// In transformers.js/src/models.js

import { PrefixConditioningVLM } from './models/prefix_conditioning.js';

// Add to MODEL_TYPES
const MODEL_TYPES = {
    // ... existing types
    PrefixConditioning: 10,
}

// Add to MODEL_CLASS_TO_NAME_MAPPING
const MODEL_CLASS_TO_NAME_MAPPING = new Map([
    // ... existing mappings
    [PrefixConditioningVLM, 'prefix_conditioning_vlm'],
]);
```

2. **Update config mappings**: Add config class to transformers.js

3. **Add to Auto classes**: Enable `AutoModel.from_pretrained()` support

## Performance Considerations

### Memory Usage

- **Vision Encoder**: ~400MB (SigLIP frozen)
- **Projection**: <1MB (Linear layer)
- **Language Model**: ~500MB (SmolLM with LoRA merged)
- **Total**: ~900MB for inference

### Inference Speed (estimated)

- **Image encoding**: ~50-100ms (SigLIP on GPU)
- **Projection**: <5ms
- **Token generation**: ~20-50ms per token (SmolLM)
- **Total caption**: ~500ms-1.5s (30 tokens, beam=3)

### Browser Compatibility

Requires:
- WebGPU or WebGL backend (via ONNX Runtime Web)
- Modern browser with WASM support
- ~1GB RAM for model loading

## Differences from Standard VLMs

### vs. LLaVA

LLaVA inserts image tokens **at specific positions** in the text:

```
[text] <image> [more text]
```

Prefix-conditioning **prepends all image tokens**:

```
[image tokens] [text]
```

This is simpler and works better for decoder-only models without cross-attention.

### vs. VisionEncoderDecoderModel

VisionEncoderDecoderModel uses:
- Encoder-decoder architecture (e.g., ViT → GPT)
- Cross-attention layers

Prefix-conditioning uses:
- Decoder-only architecture (e.g., SigLIP → SmolLM)
- No cross-attention (more efficient)

## Testing

Create a simple test to verify the implementation:

```javascript
import { PrefixConditioningVLM } from './distilvit/prefix_model.js';
import { describe, it } from 'node:test';
import assert from 'node:assert';

describe('PrefixConditioningVLM', () => {
    it('should create model with config', () => {
        const config = new PrefixConditioningConfig({
            projection_type: 'linear',
            max_length: 30,
        });
        assert.strictEqual(config.projection_type, 'linear');
        assert.strictEqual(config.max_length, 30);
    });

    it('should have required methods', () => {
        const model = new PrefixConditioningVLM(
            new PrefixConditioningConfig(),
            { vision_encoder: null, projection: null, decoder: null }
        );
        assert(typeof model.encode_image === 'function');
        assert(typeof model.generate === 'function');
        assert(typeof model.forward === 'function');
    });
});
```

## Next Steps

To complete this implementation:

1. **Export ONNX models** from the trained Python model
2. **Integrate tensor utilities** from transformers.js
3. **Complete generation loop** with beam search/sampling
4. **Test with real models** on sample images
5. **Optimize for browser** (model quantization, WebGPU)

## References

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [docs/architecture.md](architecture.md) - Python implementation details
- [docs/onnx-export-guide.md](onnx-export-guide.md) - ONNX export instructions
