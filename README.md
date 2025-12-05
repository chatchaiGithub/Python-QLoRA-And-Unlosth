# Emotional State AI - QLoRA Fine-tuning Project

Fine-tune AI models to dynamically manage emotional states through tool selection, creating an emotionally aware AI character.

## 📋 Features

- **Dynamic Emotional States**: Mood, Hunger, Love, Loneliness, Drowsiness
- **Priority System**: Drowsiness > Hunger > Love > Mood
- **Tool Selection**: AI learns to adjust emotional values based on interactions
- **1000+ Training Examples**: Diverse scenarios covering all emotional ranges
- **QLoRA Efficient Training**: 4-bit quantization for training on consumer GPUs

## 🎯 Emotional States Explained

### Mood (0-100)
- **High (80-100)**: Very trusting, willing to do anything
- **Medium (40-79)**: Normal trust levels
- **Low (0-39)**: Defensive, suspicious

### Hunger (0-100)
- **Starving (1-30)**: Extremely irritable, will prioritize food over everything
- **Hungry (31-60)**: Slightly cranky, noticeable mood impact
- **Satisfied (61-100)**: Comfortable, stable

### Love (0-100)
- **Stranger (1-20)**: Cold, formal
- **Acquaintance (21-40)**: Polite but distant
- **Friend (41-60)**: Warm, romantic feelings
- **Obsessive (61-79)**: Possessive, clingy
- **Yandere (80-100)**: ⚠️ DANGER ZONE - Extreme jealousy, unstable

### Loneliness (0-100)
- **Fine (1-30)**: Stable emotions
- **Needs Company (31-60)**: Seeks interaction
- **Desperate (61-100)**: Bipolar tendencies, emotionally unstable

### Drowsiness (0-100)
- **Alert (1-30)**: Clear thinking
- **Drowsy (31-60)**: Sluggish, unfocused
- **Incoherent (61-100)**: Barely functional, nonsensical responses

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Generate Training Data

```powershell
python generate_training_data.py
```

This creates `emotional_states_training_data_full.json` with 1000+ examples.

### 3. Train the Model

```powershell
python train_qlora.py
```

**Training Configuration:**
- Batch size: 4
- Gradient accumulation: 4 steps
- Learning rate: 2e-4
- Epochs: 3
- LoRA rank: 64
- Quantization: 4-bit (NF4)

**Hardware Requirements:**
- GPU: NVIDIA with 12GB+ VRAM (RTX 3080/4080, or better)
- RAM: 32GB+ recommended
- Storage: 50GB+ free space

**Training Time:**
- ~6-12 hours on RTX 3090
- ~4-8 hours on RTX 4090

### 4. Test the Model

Interactive mode:
```powershell
python inference.py
```

Test with examples:
```powershell
python inference.py --test
```

## 📂 Project Structure

```
QLoRA/
├── EstopianMaid-13B/          # Base model
│   ├── config.json
│   ├── model-*.safetensors
│   └── tokenizer files
├── generate_training_data.py   # Data generation script
├── train_qlora.py              # Training script
├── inference.py                # Testing/inference script
├── requirements.txt            # Python dependencies
├── emotional_states_training_data_full.json  # Generated dataset
└── emotional-qlora-output/     # Training outputs
    └── final/                  # Final trained model
```

## 💾 Dataset Structure

Each training example follows this format:

```json
{
  "instruction": "System prompt with rules...",
  "input": {
    "user_name": "Takeshi",
    "user_gender": "Male",
    "user_personality": "Kind and caring",
    "current_state": {
      "mood": 60,
      "hunger": 25,
      "love": 38,
      "loneliness": 30,
      "drowsiness": 10
    },
    "current_time": "13:00",
    "message": "Want to grab lunch?"
  },
  "output": "AI response with tool_call and emotional response"
}
```

## 🎨 Training Data Distribution

- **Hunger scenarios**: ~270 examples (starving, food offers, declines)
- **Drowsiness**: ~150 examples (late night, exhaustion)
- **Love spectrum**: ~250 examples (romantic, obsessive, yandere)
- **Loneliness**: ~100 examples (isolation, need for company)
- **Low love/criticism**: ~120 examples (strangers, conflicts)
- **Multi-crisis**: ~80 examples (multiple negative states)
- **Random varied**: ~100 examples (diverse situations)

## 🔧 Customization

### Adjust LoRA Parameters

In `train_qlora.py`:

```python
LORA_R = 64        # Higher = more parameters (better quality, slower)
LORA_ALPHA = 16    # Scaling factor
LORA_DROPOUT = 0.1 # Regularization
```

### Modify Training

```python
BATCH_SIZE = 4                      # Increase if you have more VRAM
GRADIENT_ACCUMULATION_STEPS = 4     # Effective batch size = BATCH_SIZE * this
LEARNING_RATE = 2e-4                # Adjust for stability
NUM_EPOCHS = 3                      # More epochs = better learning (risk overfitting)
```

### Add More Training Examples

Edit `generate_training_data.py`:

```python
# Add new message templates
MESSAGE_TEMPLATES = {
    "your_scenario": [
        "Template with {placeholder}",
        "Another template..."
    ]
}

# Add to scenario distribution
scenarios = [
    *[("scenario_type", "message_type") for _ in range(count)],
]
```

## 📊 Monitoring Training

Training will output:
- Loss values (should decrease)
- Evaluation metrics
- Checkpoint saves every 500 steps

Check `emotional-qlora-output/` for:
- Training logs
- Model checkpoints
- Final model

## 🎮 Using the Trained Model

### Interactive Chat

```python
from inference import load_model, generate_response

model, tokenizer = load_model()

user_input = {
    "user_name": "Alex",
    "user_gender": "Male",
    "user_personality": "Friendly",
    "current_state": {
        "mood": 70,
        "hunger": 80,
        "love": 45,
        "loneliness": 25,
        "drowsiness": 15
    },
    "current_time": "14:00",
    "message": "Hey! How are you?"
}

response = generate_response(model, tokenizer, user_input)
print(response)
```

## ⚠️ Important Notes

1. **Output Parsing**: The current dataset has placeholder outputs. You'll need to either:
   - Manually create high-quality outputs for all examples
   - Use a larger model (GPT-4, Claude) to generate outputs
   - Train iteratively and improve outputs over time

2. **State Tracking**: The inference script shows responses but doesn't automatically parse and update emotional states. You'll need to implement JSON parsing for the tool_call output.

3. **GPU Memory**: If you run out of memory:
   - Reduce `BATCH_SIZE`
   - Increase `GRADIENT_ACCUMULATION_STEPS`
   - Reduce `MAX_SEQ_LENGTH`
   - Use `load_in_8bit=True` instead of 4-bit

4. **Quality vs Speed**: 
   - More LoRA parameters = better quality but slower training
   - More epochs = better learning but risk of overfitting
   - Larger batch size = faster training but more memory

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# In train_qlora.py, reduce:
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 1536
```

### Training Too Slow
```python
# Increase batch size if you have VRAM:
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
```

### Model Not Learning
- Check if loss is decreasing
- Try increasing `LEARNING_RATE` to 3e-4
- Increase `NUM_EPOCHS` to 5
- Ensure dataset is correctly formatted

### Model Outputs Gibberish
- Likely overfitting or bad data
- Reduce epochs
- Add more diverse training data
- Check dataset quality

## 📚 Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

## 📝 License

This project is for educational and research purposes. Ensure you have the right to use and modify the base model.

## 🤝 Contributing

Feel free to:
- Add more training scenarios
- Improve output quality
- Optimize training parameters
- Create better inference tools
- Build a web UI for testing

---

**Created for training emotionally-aware AI characters with dynamic state management** 🎭✨
