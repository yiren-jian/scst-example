# SCST Training in Huggingface Transformers

## Installation
Create a conda environment for scst training
```bash
conda create -n scst-example python=3.8
conda activate scst-example
```

Download the customized Huggingface transformers (4.30.dev)
```bash
git clone git@github.com:yiren-jian/transformers.git
cd transformers
pip install -e .
pip install sentencepiece
```

Install pytorch
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## SCST training
You can try
```bash
python flan-t5-base.py
```

The reward can be computed by the following, if you have `helfulness_classifier()` preatrained.
```python
caps_gen = tokenizer.batch_decode(
    outputs.sequences, skip_special_tokens=True
)
caps_gen = [text.strip() for text in caps_gen] # a list of sentences [Bsz * num_beams]
reward = helfulness_classifier(caps_gen)
```