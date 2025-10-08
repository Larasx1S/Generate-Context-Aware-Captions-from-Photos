# Context-Aware Image Captioning with BLIP-2

**Systematic Analysis and Development of Innovative Systems 2025 - Practical Lab**  
**Project: Generate Context-Aware Captions from Photos**

This project implements a context-aware image captioning system that goes beyond standard literal descriptions. Instead of generating basic captions like "A person standing in front of a building," our system produces rich, contextual descriptions like "Reporter covering the climate summit in Paris, December 2024" by incorporating temporal, geographical, and social context.

## Project Overview

**Objective**: Develop a captioning model that incorporates contextual signals (location, people, events, dates) to generate more meaningful and informative image descriptions.

**Approach**: Fine-tune BLIP-2 on the GoodNews dataset, which contains news images with rich metadata including article context, named entities, locations, and temporal information. The model learns to integrate visual content with contextual signals.

**Context Integration Examples**:
- **Standard caption**: "A man in a suit speaking at a podium"
- **Context-aware caption**: "President Biden addressing climate change at COP28 summit in Dubai, December 2023"

**Dataset**: GoodNews - news images with articles, metadata, and rich contextual information including people, locations, events, and temporal data.

**Inspired by:** [GoodNews Repository](https://github.com/furkanbiten/GoodNews.git)


## Project Structure

```
project/
├── dataset_preparation/                # Dataset preprocessing scripts
│   ├── 0_url_to_superjumbo.py        # Convert URLs to high-resolution variants
│   ├── 1_get_images.py               # Download images from NYTimes URLs
│   ├── 2_create_context_dataset.py   # Extract context and create datasets
│   ├── 3_resize_images.py            # Resize images to 224x224 for training
│   └── 4_organize_folders.py         # Organize images into hash-based subdirectories
├── data/
│   ├── raw/                          # Original GoodNews files
│   │   ├── captioning_dataset.json   # Original captions and metadata
│   │   ├── img_urls_all.json         # Original image URLs
│   │   └── img_urls_super.json       # High-resolution URLs (generated)
│   ├── processed/                    # Processed datasets
│   │   ├── train.json                # Original format with context
│   │   ├── val.json
│   │   ├── test.json
│   │   ├── train_224.json            # Final training format
│   │   ├── val_224.json
│   │   └── test_224.json
│   └── images/
│       ├── original/                 # Downloaded images
│       └── images224/                # Resized training images
│           └── xx/                   # Hash-organized subdirectories (00-ff)
├── training/
│   ├── safe_multi_gpu_training.py    # Conservative multi-GPU training configuration
│   ├── best_caption.py               # Model inference and caption generation
│   ├── evaluate_results.py           # Evaluation metrics (BLEU, ROUGE)
│   └── run.sh                        # Script to run training
├── results/                          # Training outputs and model checkpoints
├── requirements.txt                  # Python package requirements
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- PIL (Pillow)
- spaCy (with English model: `python -m spacy download en_core_web_sm`)
- tqdm
- joblib
- bitsandbytes (for 8-bit optimization)
- CUDA-compatible GPU(s)

## Installation

1. navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Dataset Preparation

The dataset preparation involves several sequential steps to process the original GoodNews dataset:

### Step 1: Convert URLs to High Resolution
```bash
cd dataset_preparation
python 0_url_to_superjumbo.py
```

**What it does:**
- Reads `data/raw/img_urls_all.json` (original image URLs)
- Converts all image URLs to their high-resolution "superJumbo" variants
- Outputs `data/raw/img_urls_super.json`

### Step 2: Download Images
```bash
python 1_get_images.py --num_threads 8
```

**What it does:**
- Downloads images from NYTimes URLs using parallel processing
- Saves images to `data/images/original/`
- Handles network errors and retries failed downloads
- Progress tracking with thread-based downloading

**Options:**
- `--num_threads`: Number of parallel download threads (default: 4)
- `--urls_file`: Custom path to URLs JSON file
- `--output_dir`: Custom output directory for images

### Step 3: Create Context Dataset
```bash
python 2_create_context_dataset.py
```

**What it does:**
- Processes original captions and metadata from `data/raw/captioning_dataset.json`
- Extracts contextual information using spaCy NLP:
  - **Places**: Geographic locations (GPE, LOC entities)
  - **Dates**: From URLs and text (YYYY-MM-DD format)
  - **People**: Person names (PERSON, ORG entities)
  - **Events**: Major events (Olympics, World Cup, etc.)
  - **Sections**: News section from URL structure
- Creates context strings like: `[sports | New York | 2024-01-15 | PERSON=John Smith | EVENT=Olympics]`
- Outputs train/val/test splits to `data/processed/`

### Step 4: Resize Images for Training
```bash
python 3_resize_images.py
```

**What it does:**
- Resizes all images to 224×224 pixels (BLIP-2 input requirement)
- Center crops to square aspect ratio
- Converts RGBA/palette images to RGB
- Optimizes JPEG quality and file size
- Updates JSON files with new image paths
- Outputs resized images to `data/images/images224/`

### Step 5: Organize into Subdirectories
```bash
python 4_organize_folders.py
```

**What it does:**
- Organizes images into hash-based subdirectories (00-ff) for better filesystem performance
- Updates JSON file paths to reflect new directory structure
- Final format: `images224/ab/image_filename.jpg`

## Training

### Multi-GPU Distributed Training
```bash
cd training
chmod +x run.sh
./run.sh
```

The `run.sh` script automatically configures the distributed training environment:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun \
    --nproc_per_node=4 \
    --master_addr=localhost \
    --master_port=29500 \
    safe_multi_gpu_training.py
```

**Training Implementation Details** (from `safe_multi_gpu_training.py`):

**Distributed Training Setup**:
- **Backend**: NCCL for multi-GPU communication
- **Process Group**: Automatic initialization via torchrun
- **Data Distribution**: DistributedSampler ensures even data split across GPUs
- **Model Wrapping**: DistributedDataParallel (DDP) for synchronized training

**Memory Optimization**:
- **Precision**: FP16 with automatic mixed precision
- **Gradient Checkpointing**: Reduces memory at cost of computation
- **8-bit Optimizer**: bitsandbytes AdamW8bit for memory efficiency
- **Gradient Accumulation**: 64 steps to simulate large batch sizes

**Parameter Selection Logic**:
```python
# From setup_trainable_params_efficient():
last_n_layers = 1  # Configurable based on GPU memory

# Trainable components:
if "qformer" in name.lower():          # All Q-Former layers
    should_train = True
elif "lm_head" in name:               # Language model head
    should_train = True  
elif "language_model.model.decoder.layers." in name:
    layer_num = int(layer_str)
    if layer_num >= (num_decoder_layers - last_n_layers):
        should_train = True           # Last N decoder layers only
```

**Model Scaling Configuration**:
- **Current**: `model_name = "Salesforce/blip2-opt-2.7b"`
- **Alternative**: `model_name = "Salesforce/blip2-flan-t5-xl"` (requires more GPU memory)
- **Layer Scaling**: Increase `last_n_layers` with better hardware


### Hardware Scaling Guidelines

**GPU Memory Limitations:**
- Current setup uses 4x RTX 2080 Ti (11GB each) = 44GB total GPU memory
- Due to memory constraints, only `last_n_layers = 1` could be trained
- Out of memory errors occurred when attempting to train more layers
- Only **183,814,144 / 3,744,761,856 parameters (4.91%)** were trainable

**Recommended Hardware Configurations:**

| GPU Setup | Model | Trainable Layers | Expected Performance |
|-----------|-------|------------------|---------------------|
| 4x 11GB RTX 2080 Ti | blip2-opt-2.7b | `last_n_layers = 1` | Current setup |
| 4x 16GB | blip2-opt-2.7b | `last_n_layers = 4` | Better accuracy |
| 4x 24GB RTX 4090 | blip2-flan-t5-xl | `last_n_layers = 8` | Best performance |
| 8x 24GB | blip2-flan-t5-xl | Full fine-tuning | Professional grade |

**Scaling Instructions:**
1. **For larger GPUs**: Increase `last_n_layers` value in `safe_multi_gpu_training.py`
2. **For better model**: Change `model_name` from `"Salesforce/blip2-opt-2.7b"` to `"Salesforce/blip2-flan-t5-xl"`
3. **Monitor memory usage**: Use `nvidia-smi` during training to optimize layer count

## Model Inference and Testing

### Generate Captions
```bash
cd training
python best_caption.py
```

**Features:**
- Loads trained model checkpoint from `results/`
- Generates captions for test set images
- **Original caption generation only** (no name enhancement)
- Multiple parameter sets to avoid repetitive captions
- Saves results to `results/test_results_[model]_original.json`

**Caption Generation Strategy:**
- Uses multiple beam search configurations
- Repetition penalty and temperature controls
- Selects best caption with minimal word repetition
- No post-processing or name enhancement

### Evaluation

```bash
python evaluate_results.py --results_file ../results/test_results_[model]_original.json --output_file ../results/evaluation_scores.json
```

**Evaluation Metrics:**
- **BLEU-4**: Standard image captioning metric
- **ROUGE-L**: Longest common subsequence scoring
- **Length Statistics**: Reference vs. generated caption lengths
- **Caption Examples**: Sample outputs for qualitative assessment

## Training Progress & Results

### Complete Training Log (August 29-30, 2025)

The training was successfully completed over 12 epochs with the following loss progression:

```
Model: Salesforce/blip2-opt-2.7b
GPUs: 4x NVIDIA GeForce RTX 2080 Ti (11GB each)
Trainable Parameters: 183,814,144 / 3,744,761,856 (4.91%)
Strategy: Q-Former + LM Head + Last 1 OPT Decoder Layer

EPOCH  1/12: Average loss: 10.7450
EPOCH  2/12: Average loss: 1.3448
EPOCH  3/12: Average loss: 0.5968
EPOCH  4/12: Average loss: 0.5513
EPOCH  5/12: Average loss: 0.5362
EPOCH  6/12: Average loss: 0.5212
EPOCH  7/12: Average loss: 0.5136
EPOCH  8/12: Average loss: 0.5101 (Best validation: 0.4836)
EPOCH  9/12: Average loss: 0.5034
EPOCH 10/12: Average loss: 0.5054
EPOCH 11/12: Average loss: 0.4978
EPOCH 12/12: Average loss: 0.5019

Final Best Validation Loss: 0.4836
```

### Learning Rate Schedule
- **Warmup**: Steps 0-120 (linear increase to 3e-5)
- **Peak Learning**: Step 120 (3e-5)  
- **Cosine Decay**: Steps 120-600 (3e-5 → 8e-6)

### Sample Results

The model demonstrates strong context-aware caption generation:

**Example 1**

<img src="/home/labs25captioning/blip2/image_readme/28physics-jumbo.webp" alt="Text" width="500">

* **Original:** "Patriots Coach Bill Belichick on Tuesday. “I think Belichick is better at keeping pressure on the passer than passing a physics test,” a cosmologist said."
* **Generated:** "Context: [sports 2015-11-02 PERSON Bill Belichick] Caption: Bill Belichick, the coach of the New England Patriots, at a news conference in Foxborough, on Monday."
* **Context:** "[sports | Belichick | 2015-01-28 | PERSON=Bill Belichick]"

**Example 2**

<img src="/home/labs25captioning/blip2/image_readme/03erdogan-web1-superJumbo.webp" alt="Text" width="500">

* **Original:** "President Recep Tayyip Erdogan voted on Sunday at a polling station in Istanbul. His party gained a decisive victory."
* **Generated:** "Context: [world Turkey 2015-11-13 PERSON Erdogan] Caption: President Recep Tayyip Erdogan of Turkey voting in Istanbul on Sunday. Mr. Erdogans said at a news conference after casting his own vote."
* **Context:** "[world | Istanbul | 2015-11-03 | PERSON=Recep Tayyip Erdogan]"

**Example 3**


<img src="/home/labs25captioning/blip2/image_readme/04db-briefs-superJumbo.webp" alt="Text" width="500">

* **Original:** "Chief Justice John G. Roberts Jr., right, with Associate Justice Anthony M. Kennedy. Justice Roberts is known as a premier appeals brief craftsman."
* **Generated:** "Context: [us 2017-01-18 PERSON the Supreme Court] Caption: Chief Justice John G. Roberts Jr., left, and Justice Anthony M. Kennedy at the Supreme Court in Washington on Tuesday."
* **Context:** "[business | 2016-10-01 | PERSON=John G. Roberts Jr.]"


**Example 4**


<img src="/home/labs25captioning/blip2/image_readme/5ae119be068401528a2aab58_1.jpg" alt="Text" width="500">

* **Original:** "Bill Cosby arriving Wednesday at the Montgomery County Courthouse."
* **Generated:** "Context: [nyregion New York 2015-03-17 PERSON Cosby] Caption: Bill Cosby at a court hearing in Philadelphia last month. He has pleaded not guilty to charges that he sexually assaulted Andrea Constand, a former Temple University employee, in 2004."
* **Context:** "[arts | 2018-04-25 | PERSON=the Montgomery County Courthouse]"


**Example 5**


<img src="/home/labs25captioning/blip2/image_readme/59971b9e95d0e0246f2006fa_0.jpg" alt="Text" width="500">

* **Original:** "Mourners on Friday on Las Ramblas, the site of a deadly terrorist attack in Barcelona, Spain."
* **Generated:** "Context: [world Barcelona 2017-09-18 PERSON Marjory Stoneman Douglas High School] Caption: A memorial for the victims of the attack in Barcelona, Spain, on Wednesday."
* **Context:** "[world | Las Ramblas | 2017-08-19]"


### Performance Characteristics
- **Context Integration**: Successfully incorporates location, date, and entity information
- **Temporal Accuracy**: Generates appropriate time references from context
- **Entity Recognition**: Identifies and includes relevant people and organizations
- **Event Understanding**: Captures major events and competitions (World Cup, Champions League)
- **Geographic Awareness**: Properly contextualizes location information

### Memory Usage & Limitations
- **GPU Memory**: 4x 11GB = 44GB total
- **Memory Constraint**: Limited to 1 decoder layer (`last_n_layers = 1`)
- **Out-of-Memory**: Encountered when trying to train more layers
- **Solution**: Use larger GPUs (24GB) for more trainable layers

## Implementation Notes

### Model Architecture
- **BLIP-2**: Two-stage bootstrapped training approach
- **Vision Encoder**: Frozen pre-trained image encoder
- **Q-Former**: Trainable module bridging vision and language (188M params)
- **Language Model**: OPT-2.7b with selective fine-tuning

### Training Strategy
- **Conservative Approach**: Minimal trainable parameters to prevent overfitting
- **Gradient Checkpointing**: Memory optimization for large model training
- **8-bit Optimization**: Reduces memory usage with minimal performance loss
- **Distributed Training**: Efficient multi-GPU utilization

### Data Processing
- **Context Integration**: News-specific metadata enhancement
- **Named Entity Recognition**: Automatic extraction of people, places, events
- **Image Preprocessing**: Standard vision model input preparation
- **Hash-based Organization**: Scalable filesystem management

### Challenges Encountered

1. **Memory Management**: Large model requires careful memory optimization
2. **Dataset Scale**: 16k samples balanced for training efficiency
3. **Context Integration**: NLP pipeline for metadata extraction
4. **Multi-GPU Coordination**: Distributed training synchronization
5. **Evaluation Metrics**: Standardized comparison with original work


## Citation

If you use this code or approach, please cite the original BLIP-2 paper:
```
@article{li2023blip2,
  title={BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author={Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},
  journal={arXiv preprint arXiv:2301.12597},
  year={2023}
}
```


## Contact

Ali Shariati Najafabadi  
alishariaty0854@gmail.com
