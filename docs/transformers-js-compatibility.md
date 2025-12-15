# Transformers.js Compatibility Analysis

## Summary

❌ **The new prefix-conditioning + LoRA architecture (PrefixConditioningVLM) is NOT directly supported by transformers.js**

✅ **The old cross-attention architecture (VisionEncoderDecoderModel) IS supported by transformers.js**

## Current Transformers.js Support (v3, 2024-2025)

### Supported Vision-Language Models

Based on the [official transformers.js documentation](https://huggingface.co/docs/transformers.js/en/api/models):

**Directly Supported:**
- ✅ `VisionEncoderDecoderModel` - Generic vision encoder-decoder (the OLD architecture)
- ✅ `SmolVLMForConditionalGeneration` - SmolVLM with SigLIP vision encoder
- ✅ `LlavaForConditionalGeneration` - LLAVA vision-language model
- ✅ `Idefics3ForConditionalGeneration` - Idefics3 multimodal model

**Vision Components:**
- ✅ `SiglipModel`, `SiglipTextModel`, `SiglipVisionModel` - SigLIP is fully supported
- ✅ `CLIPModel`, `CLIPVisionModel` - CLIP models supported

**Language Models:**
- ✅ GPT2, GPT-Neo, GPT-J (used in old architecture)
- ❌ SmolLM - NOT explicitly listed (though similar Llama-based models exist)
- ✅ T5, BART, OPT, Falcon, and many others

### Custom Architecture Support

❌ **Custom architectures are NOT supported** - Transformers.js uses a predefined class mapping system and doesn't support loading custom model types that aren't in the library.

## Compatibility Analysis

### Old Architecture (Cross-Attention + GPT2)

✅ **FULLY COMPATIBLE**

```javascript
import { pipeline } from '@huggingface/transformers';

// This will work - VisionEncoderDecoderModel is supported
const captioner = await pipeline(
  'image-to-text',
  './siglip-base-patch16-224-gpt2'
);

const result = await captioner('image.jpg');
console.log(result[0].generated_text);
```

**Components:**
- ✅ VisionEncoderDecoderModel (supported)
- ✅ SigLIP encoder (supported)
- ✅ GPT2 decoder (supported)

### New Architecture (Prefix-Conditioning + LoRA)

❌ **NOT COMPATIBLE** - Requires custom implementation

**Issues:**
1. ❌ `PrefixConditioningVLM` is a custom class not in transformers.js
2. ❌ LoRA adapters via PEFT library not supported in transformers.js
3. ❌ Custom forward pass logic (prefix projection) not recognized

**What transformers.js expects:**
- Standard HuggingFace model types from their class registry
- ONNX-converted models with standard inputs/outputs
- No custom model architectures without library modification

## Recommended Solutions

### Option 1: Use Old Architecture for Transformers.js

If transformers.js compatibility is required, continue using the old architecture:

```bash
# Train with old architecture (cross-attention)
bin/train --dataset flickr \
          --encoder-model google/siglip-base-patch16-224 \
          --decoder-model openai-community/gpt2 \
          --epochs 5
```

**Trade-offs:**
- ✅ Direct transformers.js support
- ✅ No custom code needed
- ❌ Slower training (5.7x)
- ❌ 100% parameters trainable
- ❌ Can't use modern LLMs (SmolLM, Qwen2, Phi)

### Option 2: Export New Model to ONNX + Custom Runtime

Export the new model to ONNX and write custom inference code:

```python
# Export to ONNX
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="./siglip-base-patch16-224-SmolLM-135M-lora",
    output="./onnx_model/",
    task="image-to-text"
)
```

Then use ONNX Runtime Web:

```javascript
// Custom inference with ONNX Runtime Web
import * as ort from 'onnxruntime-web';

// Load ONNX model
const session = await ort.InferenceSession.create('./onnx_model/model.onnx');

// Custom preprocessing and inference
// (requires manual implementation of vision preprocessing + generation loop)
```

**Trade-offs:**
- ✅ Can use new architecture
- ✅ ONNX optimizations
- ❌ Requires custom preprocessing/postprocessing code
- ❌ No high-level pipeline API
- ❌ More complex to maintain

### Option 3: Contribute to Transformers.js

Add PrefixConditioningVLM support to transformers.js:

1. Fork [huggingface/transformers.js](https://github.com/huggingface/transformers.js)
2. Add model class to `src/models.js`
3. Implement forward pass in JavaScript
4. Add ONNX conversion support
5. Submit pull request

**Trade-offs:**
- ✅ Community benefit
- ✅ Native transformers.js support
- ❌ Significant development effort
- ❌ Requires JavaScript/TypeScript expertise
- ❌ PR review and merge timeline

### Option 4: Use Similar Supported Model (SmolVLM)

Use SmolVLM which has similar architecture (SigLIP + LM):

```javascript
import { pipeline } from '@huggingface/transformers';

// SmolVLM is supported and uses SigLIP + language model
const captioner = await pipeline(
  'image-to-text',
  'HuggingFaceTB/SmolVLM-256M-Instruct'
);
```

**Trade-offs:**
- ✅ Fully supported in transformers.js
- ✅ Similar architecture concept
- ❌ Not your trained model
- ❌ Would need to retrain if you want custom behavior

## Performance Considerations

### Old Architecture (Browser-Compatible)

**Pros:**
- Direct transformers.js pipeline support
- WebGPU acceleration available
- Proven browser deployment

**Cons:**
- Larger model size to load in browser
- Slower inference (full model weights)

### New Architecture (Server/Native Preferred)

**Pros:**
- 99% smaller weights to load (LoRA adapters only)
- Faster inference
- Better quality (modern LLMs)

**Cons:**
- Not browser-compatible without custom work
- Better suited for server/native deployment

## Recommendation

**If transformers.js/browser deployment is a requirement:**
- ✅ **Use the old VisionEncoderDecoderModel architecture** (fully supported)
- Train with: `--encoder-model google/siglip-base-patch16-224 --decoder-model openai-community/gpt2`

**If server/native deployment is acceptable:**
- ✅ **Use the new PrefixConditioningVLM architecture** (5.7x faster training, better quality)
- Deploy via Python/ONNX Runtime on server
- Use API for browser access

**Hybrid approach:**
- Train with new architecture for efficiency
- Fine-tune a VisionEncoderDecoderModel for browser deployment
- Use new model on server, old model in browser as fallback

## Conclusion

The new prefix-conditioning + LoRA architecture prioritizes training efficiency and modern LLM usage over browser compatibility. For transformers.js deployment, the old architecture remains the recommended choice unless you're willing to invest in custom ONNX export and inference code.

---

**Sources:**
- [Transformers.js Documentation - Models](https://huggingface.co/docs/transformers.js/en/api/models)
- [Transformers.js v3 Announcement](https://huggingface.co/blog/transformersjs-v3)
- [Transformers.js GitHub](https://github.com/huggingface/transformers.js)
- [NPM Package](https://www.npmjs.com/package/@xenova/transformers)
