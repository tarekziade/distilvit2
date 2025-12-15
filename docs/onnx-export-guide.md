# ONNX Export Guide: Merging LoRA and Static Weights

## Overview

This guide explains how to **merge LoRA adapters into static weights** and export to ONNX for browser deployment, potentially enabling transformers.js compatibility or custom ONNX Runtime Web inference.

## Strategy

```
Trained Model (LoRA adapters)
    ↓ merge_and_unload()
Static Weight Model (no adapters)
    ↓ torch.onnx.export()
ONNX Model (self-contained graph)
    ↓ ONNX Runtime Web
Browser Inference
```

## Step 1: Merge LoRA Weights

The PEFT library provides `merge_and_unload()` to merge LoRA adapters back into base model weights:

```python
#!/usr/bin/env python3
"""
Merge LoRA weights into base model and save as standard model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import SiglipVisionModel
from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig
import json

def merge_lora_weights(model_path, output_path):
    """Merge LoRA adapters into base model weights."""

    # Load config
    with open(f"{model_path}/config.json", "r") as f:
        config_dict = json.load(f)
    config = PrefixConditioningConfig(**config_dict)

    # Load base models
    vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # Create model and load trained weights
    model = PrefixConditioningVLM(config, vision_encoder, language_model)
    from safetensors.torch import load_file
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    # Merge LoRA weights into language model
    print("Merging LoRA adapters...")
    merged_language_model = model.language_model.merge_and_unload()

    # Create new model with merged weights
    merged_model = PrefixConditioningVLM(
        config,
        model.vision_encoder,  # Vision encoder unchanged (was frozen)
        merged_language_model   # Language model with LoRA merged
    )
    merged_model.projection = model.projection  # Projection was fully trained

    # Save merged model
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)

    print("✓ LoRA weights merged successfully!")
    return merged_model

if __name__ == "__main__":
    model_path = "./siglip-base-patch16-224-SmolLM-135M-lora"
    output_path = "./siglip-base-patch16-224-SmolLM-135M-merged"

    merge_lora_weights(model_path, output_path)
```

## Step 2: Export to ONNX

Export the merged model to ONNX format:

```python
#!/usr/bin/env python3
"""
Export merged model to ONNX format.
"""
import torch
import torch.onnx
from transformers import AutoImageProcessor
from PIL import Image
import numpy as np

def export_to_onnx(model_path, output_path):
    """Export model to ONNX with static weights."""

    # Load merged model
    from distilvit.prefix_model import PrefixConditioningVLM, PrefixConditioningConfig
    from transformers import AutoModelForCausalLM
    from transformers import SiglipVisionModel
    import json

    with open(f"{model_path}/config.json", "r") as f:
        config_dict = json.load(f)
    config = PrefixConditioningConfig(**config_dict)

    vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")

    model = PrefixConditioningVLM(config, vision_encoder, language_model)
    from safetensors.torch import load_file
    state_dict = load_file(f"{model_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Create dummy input for tracing
    image_processor = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    dummy_image = Image.new('RGB', (224, 224), color='red')
    pixel_values = image_processor(dummy_image, return_tensors="pt").pixel_values

    print("Exporting to ONNX...")
    print(f"Input shape: {pixel_values.shape}")

    # Export with torch.onnx.export
    torch.onnx.export(
        model,
        (pixel_values,),
        f"{output_path}/model.onnx",
        input_names=['pixel_values'],
        output_names=['generated_ids'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'generated_ids': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"✓ Model exported to {output_path}/model.onnx")

    # Verify export
    import onnx
    onnx_model = onnx.load(f"{output_path}/model.onnx")
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified!")

if __name__ == "__main__":
    model_path = "./siglip-base-patch16-224-SmolLM-135M-merged"
    output_path = "./onnx_export"

    export_to_onnx(model_path, output_path)
```

## Step 3: Browser Inference with ONNX Runtime Web

### Option A: Direct ONNX Runtime Web (Recommended)

Since transformers.js doesn't support custom architectures, use ONNX Runtime Web directly:

```javascript
import * as ort from 'onnxruntime-web';

class VisionLanguageModel {
  constructor() {
    this.session = null;
    this.imageProcessor = null;
  }

  async load(modelPath) {
    // Load ONNX model
    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgpu', 'wasm']
    });

    console.log('Model loaded successfully');
  }

  async preprocessImage(imageUrl) {
    // Load image
    const img = new Image();
    img.src = imageUrl;
    await img.decode();

    // Create canvas for preprocessing
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 224, 224);

    // Get pixel data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const pixels = imageData.data;

    // Convert to normalized tensor [1, 3, 224, 224]
    const float32Data = new Float32Array(1 * 3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
      // Normalize to [-1, 1] (adjust based on model's normalization)
      float32Data[i] = (pixels[i * 4] / 255 - 0.5) / 0.5;           // R
      float32Data[224 * 224 + i] = (pixels[i * 4 + 1] / 255 - 0.5) / 0.5; // G
      float32Data[224 * 224 * 2 + i] = (pixels[i * 4 + 2] / 255 - 0.5) / 0.5; // B
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  async generateCaption(imageUrl) {
    const pixelValues = await this.preprocessImage(imageUrl);

    // Run inference
    const feeds = { pixel_values: pixelValues };
    const results = await this.session.run(feeds);

    // Decode output tokens (requires tokenizer logic)
    const generatedIds = results.generated_ids.data;

    return this.decodeTokens(generatedIds);
  }

  decodeTokens(tokenIds) {
    // TODO: Implement tokenizer decode logic
    // This requires porting the tokenizer or using a JS tokenizer
    return "Caption: [Decoded text]";
  }
}

// Usage
const model = new VisionLanguageModel();
await model.load('./onnx_export/model.onnx');
const caption = await model.generateCaption('image.jpg');
console.log(caption);
```

### Option B: Wrapper for Transformers.js Pipeline

Create a custom pipeline wrapper:

```javascript
import { pipeline, AutoTokenizer } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

class CustomVisionLanguagePipeline {
  async init(onnxModelPath, tokenizerPath) {
    this.onnxSession = await ort.InferenceSession.create(onnxModelPath);
    this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath);
  }

  async __call__(image) {
    // Preprocess image
    const pixelValues = await this.preprocessImage(image);

    // Run ONNX inference
    const results = await this.onnxSession.run({ pixel_values: pixelValues });

    // Decode with HF tokenizer
    const text = this.tokenizer.decode(results.generated_ids.data, {
      skip_special_tokens: true
    });

    return [{ generated_text: text }];
  }

  async preprocessImage(image) {
    // Same as above
  }
}
```

## Challenges and Solutions

### Challenge 1: Custom Architecture Not in Transformers.js

**Problem:** PrefixConditioningVLM is not a standard transformers.js model class.

**Solution:**
- ✅ ONNX export captures the entire forward pass graph
- ✅ ONNX model is self-contained with all operations
- ✅ Use ONNX Runtime Web directly (bypass transformers.js)

### Challenge 2: Generation Loop

**Problem:** ONNX export captures one forward pass, not the autoregressive generation loop.

**Solutions:**

**Option A: Export with generation loop (complex):**
```python
# Create a wrapper that includes generation
class ModelWithGeneration(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        # Implement full generation loop in forward
        return self.model.generate(pixel_values, max_length=30)

# Export wrapper
wrapper = ModelWithGeneration(model)
torch.onnx.export(wrapper, ...)
```

**Option B: Implement generation in JavaScript (simpler):**
```javascript
async function generateText(session, pixelValues, maxLength = 30) {
  let generatedIds = [];

  for (let i = 0; i < maxLength; i++) {
    // Run one forward pass
    const outputs = await session.run({
      pixel_values: pixelValues,
      input_ids: new ort.Tensor('int64', generatedIds, [1, generatedIds.length])
    });

    // Get next token
    const nextToken = getNextToken(outputs.logits);
    generatedIds.push(nextToken);

    // Check for EOS
    if (nextToken === eosTokenId) break;
  }

  return generatedIds;
}
```

### Challenge 3: Model Size

**Problem:** Full model (~900MB) is large for browser loading.

**Solutions:**
- ✅ Quantize to INT8 or FP16 (reduces to ~230MB)
- ✅ Use dynamic quantization during export
- ✅ Lazy load model chunks
- ✅ Cache model in IndexedDB

```python
# Quantize during export
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

### Challenge 4: Tokenizer in JavaScript

**Problem:** Need tokenizer for decoding in browser.

**Solutions:**
- ✅ Use transformers.js tokenizer (compatible)
- ✅ Export tokenizer.json (HF format)
- ✅ Use @huggingface/transformers AutoTokenizer

```javascript
import { AutoTokenizer } from '@huggingface/transformers';

const tokenizer = await AutoTokenizer.from_pretrained(
  './siglip-base-patch16-224-SmolLM-135M-merged'
);

const text = tokenizer.decode(generatedIds, { skip_special_tokens: true });
```

## Performance Comparison

| Approach | Model Size | Load Time | Inference Speed | Compatibility |
|----------|-----------|-----------|----------------|---------------|
| **New + ONNX (FP32)** | ~900MB | ~5-10s | Fast (WASM/WebGPU) | Custom code |
| **New + ONNX (INT8)** | ~230MB | ~2-3s | Very fast | Custom code |
| **Old + Transformers.js** | ~400MB | ~3-5s | Fast | Native support |

## Complete Workflow

```bash
# 1. Train with new architecture
make train DATASET=flickr EPOCHS=5

# 2. Merge LoRA weights
python merge_lora.py

# 3. Export to ONNX
python export_onnx.py

# 4. Quantize (optional but recommended)
python -m onnxruntime.quantization.preprocess \
    --input model.onnx \
    --output model_quantized.onnx

# 5. Deploy to web
cp onnx_export/* ./web/models/
```

## Conclusion

### ✅ This Approach Works!

Merging LoRA and exporting to ONNX **is viable** for browser deployment:

**Pros:**
- ✅ Static weights (no LoRA adapters needed at runtime)
- ✅ ONNX captures full forward pass graph
- ✅ WebGPU/WASM acceleration available
- ✅ Can quantize to reduce size (4x smaller)
- ✅ Self-contained model file

**Cons:**
- ❌ Not plug-and-play with transformers.js pipeline
- ❌ Requires custom preprocessing/postprocessing code
- ❌ Generation loop needs JavaScript implementation
- ❌ More maintenance overhead

### Recommendation

**If you need browser deployment:**
1. **Quick & Easy**: Use old VisionEncoderDecoderModel + transformers.js (native support)
2. **Best Performance**: Use new architecture + ONNX export (this approach)

**If you choose ONNX route:**
- Expect 1-2 weeks of additional development for custom inference code
- Budget for testing/debugging browser compatibility
- Consider hiring frontend developer familiar with ONNX Runtime Web

---

**Sources:**
- [PEFT merge_and_unload Documentation](https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968)
- [Export PEFT/LoRA to ONNX Issue](https://github.com/huggingface/peft/issues/670)
- [Merging LoRA Weights Guide](https://apxml.com/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/merging-lora-weights)
- [PyTorch ONNX Export Tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
- [ONNX Runtime Export Guide](https://onnxruntime.ai/docs/tutorials/export-pytorch-model.html)
- [torch.onnx Documentation](https://docs.pytorch.org/docs/stable/onnx.html)
