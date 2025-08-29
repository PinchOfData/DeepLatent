# Image Support in DeepLatent

This document describes how to use the image support functionality added to the DeepLatent framework.

## Overview

The framework now supports images as a modality alongside text, embeddings, votes, and discrete choices. Images are processed using:

- **CNN-based encoders** with FiLM (Feature-wise Linear Modulation) for covariate conditioning
- **CNN-based decoders** for image reconstruction
- **Lazy loading** to handle large image datasets efficiently
- **Multimodal fusion** with text and other modalities

## Key Features

### 1. FiLM Conditioning
Images can be conditioned on covariates (like document metadata) using FiLM layers that apply affine transformations to CNN feature maps based on covariate values.

### 2. Lazy Loading
Images are loaded and transformed on-demand during training, making it possible to work with large datasets that don't fit in memory.

### 3. Flexible Image Processing
Users provide their own transformation functions, allowing complete control over image preprocessing, augmentation, and feature extraction.

## Usage

### 1. Define Image Transform Function

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

def image_transform_function(image_paths):
    \"\"\"Custom image transformation function for lazy loading.\"\"\"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    processed_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            tensor = transform(img)
            processed_images.append(tensor)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            processed_images.append(torch.zeros(3, 224, 224))  # Fallback
    
    # For lazy loading, return single tensor if only one image
    if len(processed_images) == 1:
        return processed_images[0]
    return torch.stack(processed_images)
```

### 2. Configure Corpus with Images

```python
from src.corpus import Corpus
import pandas as pd

# Your DataFrame should have a column with image paths
df = pd.DataFrame({
    'text': ["Sample text 1", "Sample text 2"],
    'image_path': ["/path/to/image1.jpg", "/path/to/image2.jpg"],
    'category': ['A', 'B']
})

modalities = {
    "text": {
        "views": {
            "bow": {
                "type": "bow",
                "column": "text",
                "vectorizer": CountVectorizer(max_features=1000)
            }
        }
    },
    "visual": {
        "views": {
            "raw_pixels": {
                "type": "image",
                "column": "image_path",
                "transform_fn": image_transform_function  # Your transform function
            }
        }
    }
}

corpus = Corpus(df, modalities=modalities, prevalence="~ category")
```

### 3. Configure Image Encoders and Decoders

```python
# Image encoder configuration
encoder_args = {
    "visual_raw_pixels": {
        "input_channels": 3,
        "hidden_dims": [32, 64, 128, 256],    # CNN layers
        "fc_hidden_dims": [512],              # FC layers after CNN
        "activation": "relu",
        "dropout": 0.1,
        "use_batch_norm": True
    },
    "text_bow": {
        "hidden_dims": [512, 256],
        "activation": "relu"
    }
}

# Image decoder configuration  
decoder_args = {
    "visual_raw_pixels": {
        "output_channels": 3,
        "hidden_dims": [256, 128, 64, 32],    # Deconv layers
        "output_size": (224, 224),
        "fc_hidden_dims": [512],
        "activation": "relu",
        "use_batch_norm": True
    },
    "text_bow": {
        "hidden_dims": [256, 512]
    }
}
```

### 4. Train Multimodal Model

```python
from src.models import GTM

model = GTM(
    train_data=corpus,
    n_factors=10,
    encoder_args=encoder_args,
    decoder_args=decoder_args,
    fusion="moe_average",  # or "moe_gating", "poe"
    batch_size=16,         # Smaller batches for images
    num_epochs=100
)

model.train(corpus)

# Get results
doc_topics = model.get_doc_topic_distribution(corpus)
topic_words = model.get_topic_words()
```

## Architecture Details

### ImageEncoder
- CNN feature extraction with configurable channel progression
- FiLM layers for covariate conditioning after each CNN block
- Final FC layers to map to latent space
- Supports prevalence covariates and labels separately

### ImageDecoder  
- FC layers to expand from latent space
- Deconvolutional layers for upsampling
- Sigmoid activation for [0,1] output range
- Interpolation to exact output size if needed

### Fusion Methods
All existing fusion methods work with images:
- **MoE Average**: Simple averaging of modality outputs
- **MoE Gating**: Learned attention weights over modalities  
- **PoE**: Product of Experts for VAE posteriors

## Performance Considerations

1. **Memory**: Images use much more memory than text. Use smaller batch sizes (8-32).

2. **Lazy Loading**: Images are loaded during training, which can be slower than pre-loading but saves memory.

3. **GPU**: Image processing benefits significantly from GPU acceleration.

4. **Transform Function**: Optimize your transform function for your specific use case. Consider:
   - Caching preprocessed images
   - Parallel image loading
   - Data augmentation for training

## Example Applications

- **Multimodal Topic Modeling**: Find topics that span both visual and textual content
- **Document Understanding**: Process documents with both text and images  
- **Social Media Analysis**: Analyze posts with text and associated images
- **Scientific Literature**: Process papers with text and figures

See `example_image_usage.py` for a complete working example.
