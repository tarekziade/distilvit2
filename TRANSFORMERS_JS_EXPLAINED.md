# Transformers.js: Plug-and-Play vs Custom Implementation

## What "Plug-and-Play" Means

### ✅ Plug-and-Play Example (Old Architecture)

With the **old VisionEncoderDecoderModel**, this just works:

```javascript
import { pipeline } from '@huggingface/transformers';

// ONE LINE - Load model
const captioner = await pipeline(
  'image-to-text',
  './siglip-base-patch16-224-gpt2'
);

// ONE LINE - Generate caption
const result = await captioner('image.jpg');
console.log(result[0].generated_text);
// Output: "a woman smiling at the camera"
```

**Why it works:**
- ✅ `VisionEncoderDecoderModel` is in transformers.js registry
- ✅ Image preprocessing is handled automatically
- ✅ Generation loop is handled automatically
- ✅ Token decoding is handled automatically
- ✅ No configuration needed

### ❌ NOT Plug-and-Play (New Architecture with ONNX)

With **PrefixConditioningVLM exported to ONNX**, this FAILS:

```javascript
import { pipeline } from '@huggingface/transformers';

// This will throw an error!
const captioner = await pipeline(
  'image-to-text',
  './siglip-base-patch16-224-SmolLM-135M-merged'
);
// ❌ Error: Unknown model type 'prefix_conditioning_vlm'
```

**Why it fails:**
- ❌ `PrefixConditioningVLM` not in transformers.js registry
- ❌ Custom architecture not recognized
- ❌ No automatic preprocessing for custom model
- ❌ No automatic generation loop
- ❌ Pipeline API doesn't know how to handle it

## What You Have To Implement Manually

### The "Plug-and-Play" Pipeline Does This Automatically:

```javascript
// What transformers.js pipeline does for you (pseudocode):
async function pipeline(task, modelId) {
  // 1. Load config and determine model type
  const config = await fetch(`${modelId}/config.json`);
  const ModelClass = MODEL_REGISTRY[config.model_type]; // ← Looks up model class

  // 2. Load preprocessor for the task
  const preprocessor = await loadPreprocessor(modelId, task);

  // 3. Load model weights
  const model = await ModelClass.from_pretrained(modelId);

  // 4. Return a callable function
  return async (input) => {
    const processed = await preprocessor(input);     // Auto preprocessing
    const outputs = await model.generate(processed); // Auto generation
    return postprocess(outputs);                     // Auto decoding
  };
}
```

### With Custom ONNX Export, You Must Implement:

```javascript
// What YOU have to write for custom ONNX model:
import * as ort from 'onnxruntime-web';
import { AutoTokenizer, AutoImageProcessor } from '@huggingface/transformers';

class CustomImageCaptioner {
  async init(onnxPath, tokenizerPath) {
    // 1. Manually load ONNX session
    this.session = await ort.InferenceSession.create(onnxPath, {
      executionProviders: ['webgpu', 'wasm']
    });

    // 2. Manually load tokenizer
    this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);

    // 3. Manually load image processor
    this.imageProcessor = await AutoImageProcessor.from_pretrained(
      'google/siglip-base-patch16-224'
    );
  }

  async preprocessImage(imagePath) {
    // 4. Manually implement image preprocessing
    const img = await loadImage(imagePath);
    const canvas = createCanvas(224, 224);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 224, 224);

    // 5. Manually normalize pixels
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const float32Data = new Float32Array(3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
      // SigLIP normalization
      float32Data[i] = (imageData.data[i * 4] / 255 - 0.5) / 0.5;
      float32Data[224 * 224 + i] = (imageData.data[i * 4 + 1] / 255 - 0.5) / 0.5;
      float32Data[224 * 224 * 2 + i] = (imageData.data[i * 4 + 2] / 255 - 0.5) / 0.5;
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  async generate(pixelValues, maxLength = 30) {
    // 6. Manually implement generation loop
    const generatedIds = [];

    for (let step = 0; step < maxLength; step++) {
      // Run one forward pass
      const feeds = {
        pixel_values: pixelValues,
        input_ids: new ort.Tensor('int64', Int64Array.from(generatedIds), [1, generatedIds.length])
      };

      const outputs = await this.session.run(feeds);

      // Get logits for next token
      const logits = outputs.logits.data;
      const vocabSize = outputs.logits.dims[2];
      const lastTokenLogits = logits.slice(-vocabSize);

      // Apply sampling/greedy decoding
      const nextTokenId = this.selectNextToken(lastTokenLogits);
      generatedIds.push(nextTokenId);

      // Check for end-of-sequence
      if (nextTokenId === this.tokenizer.eos_token_id) break;
    }

    return generatedIds;
  }

  selectNextToken(logits) {
    // 7. Manually implement token selection (greedy or sampling)
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  async caption(imagePath) {
    // 8. Manually orchestrate the full pipeline
    const pixelValues = await this.preprocessImage(imagePath);
    const generatedIds = await this.generate(pixelValues);
    const text = this.tokenizer.decode(generatedIds, { skip_special_tokens: true });
    return text;
  }
}

// Usage (more verbose than pipeline)
const captioner = new CustomImageCaptioner();
await captioner.init('./model.onnx', './tokenizer');
const caption = await captioner.caption('image.jpg');
```

## Complexity Comparison

### Lines of Code

| Approach | Your Code | What It Does |
|----------|-----------|-------------|
| **Plug-and-Play** | ~5 lines | Everything (load, preprocess, generate, decode) |
| **Custom ONNX** | ~200+ lines | You implement everything yourself |

### Components You Must Implement

| Component | Plug-and-Play | Custom ONNX |
|-----------|---------------|-------------|
| Model Loading | ✅ Automatic | ❌ Manual (ONNX Runtime Web) |
| Image Preprocessing | ✅ Automatic | ❌ Manual (canvas + normalization) |
| Tensor Creation | ✅ Automatic | ❌ Manual (Float32Array) |
| Generation Loop | ✅ Automatic | ❌ Manual (autoregressive sampling) |
| Token Selection | ✅ Automatic | ❌ Manual (greedy/sampling logic) |
| Decoding | ✅ Automatic | ❌ Manual (tokenizer.decode) |
| Error Handling | ✅ Built-in | ❌ You implement |
| Batching | ✅ Supported | ❌ You implement |
| Streaming | ✅ Supported | ❌ You implement |

## Why Transformers.js Can't Load Custom Models

### Model Registry System

Transformers.js uses a model registry to map model types to classes:

```javascript
// Inside transformers.js source code
const MODEL_CLASS_MAPPINGS = {
  'vision-encoder-decoder': VisionEncoderDecoderModel,
  'clip': CLIPModel,
  'siglip': SiglipModel,
  'gpt2': GPT2LMHeadModel,
  'llava': LlavaForConditionalGeneration,
  // ... 120+ model types
  // ❌ 'prefix_conditioning_vlm' is NOT here
};

// When you call pipeline() or AutoModel.from_pretrained()
async function from_pretrained(modelId) {
  const config = await loadConfig(modelId);
  const ModelClass = MODEL_CLASS_MAPPINGS[config.model_type];

  if (!ModelClass) {
    throw new Error(`Unknown model type: ${config.model_type}`);
  }

  return await ModelClass.load(modelId);
}
```

### Your Model's config.json

```json
{
  "model_type": "prefix_conditioning_vlm",  // ← Not in registry!
  "architectures": ["PrefixConditioningVLM"],
  // ...
}
```

When transformers.js sees this, it throws:
```
Error: Unknown model type 'prefix_conditioning_vlm'
```

### What Would Be Needed for Plug-and-Play

To make it work with transformers.js, you'd need to:

1. **Add to transformers.js source code:**
```javascript
// In transformers.js/src/models.js
export class PrefixConditioningVLM extends PreTrainedModel {
  constructor(config) {
    // Implement model in JavaScript
  }

  async forward(inputs) {
    // Implement forward pass in JavaScript
  }

  async generate(inputs, options) {
    // Implement generation in JavaScript
  }
}

// Register in MODEL_CLASS_MAPPINGS
MODEL_CLASS_MAPPINGS['prefix_conditioning_vlm'] = PrefixConditioningVLM;
```

2. **Submit PR to transformers.js repository**
3. **Wait for review and merge**
4. **Wait for new release**
5. **Users update to new version**

This is **months of work** and requires:
- Deep JavaScript/TypeScript expertise
- Understanding of transformers.js internals
- Community contribution process

## The Alternative: Direct ONNX Runtime

Since custom model classes aren't feasible, use ONNX Runtime Web directly:

### Pros:
- ✅ Works with ANY ONNX model (custom or not)
- ✅ WebGPU acceleration
- ✅ No dependency on transformers.js model registry
- ✅ Full control over inference

### Cons:
- ❌ Must write your own preprocessing/generation code
- ❌ More verbose API
- ❌ No high-level pipeline abstraction

## Summary

### "Plug-and-Play" Means:

```javascript
// 2 lines, zero configuration
const captioner = await pipeline('image-to-text', 'model-id');
const result = await captioner('image.jpg');
```

### "Custom Implementation" Means:

```javascript
// 200+ lines of preprocessing, generation, and decoding
class CustomCaptioner {
  // ... implement everything yourself
}
```

### Why ONNX Export Is Not Plug-and-Play:

1. ❌ **Custom model type** not in transformers.js registry
2. ❌ **No automatic preprocessing** for your model
3. ❌ **No automatic generation** for your architecture
4. ❌ **Must use ONNX Runtime Web** directly (not transformers.js)
5. ❌ **Must implement** all pipeline logic yourself

### Bottom Line:

- **Old architecture**: Use transformers.js pipeline (5 lines, works today)
- **New architecture**: Build custom inference (200+ lines, 1-2 weeks work)

The ONNX export is **technically viable** but requires **significant custom code** instead of using the convenient high-level API.
