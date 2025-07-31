# SolarGPT

LLM-powered chatbot focused on heliophysics

### Dependencies

We used a Linux server equipped with an NVIDIA GPU, which has Python Version of 3.12.3, with following required packages to run our models:

- torch==2.6.0+cu124
- scikit-learn==1.6.1
- transformers==4.48.2
- datasets==3.3.2
- peft==0.15.2
- bitsandbytes==0.45.5
- accelerate==1.6.0

### Hugging Face Llama 3 Access

**Note:** Access to the Hugging Face Llama 3 models requires approval from Hugging Face and a valid authentication token. To gain access:

1. Request access on the [Llama-3.1-8B model card](https://huggingface.co/meta-llama/Llama-3.1-8B).
2. Once approved, export your token:
   ```bash
   export HUGGINGFACE_HUB_TOKEN="<your_token_here>"
   ```
3. Verify you can pull the model:
   ```bash
   huggingface-cli ls | grep Llama-3.1-8B
   ```

### Instructions

**Step 1:** Install dependencies using terminal:

```bash
pip install -r requirements.txt
```

**Step 2:** Ensure your `HUGGINGFACE_HUB_TOKEN` is set in your environment.\
**Step 3:** Run the chatbot:

```bash
python qa_lora3_prompt_tuning.py
```

