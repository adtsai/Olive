# BERT optimization with Intel® Neural Compressor PTQ on CPU
This folder contains a sample use case of Olive to optimize a ResNet model using onnx conversion and Intel® Neural Compressor quantization tuner.

Performs optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Intel® Neural Compressor Quantized Onnx Model

Outputs the best metrics, model, and corresponding Olive config.

## Prerequisites
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Run sample
### Run Intel® Neural Compressor quantization with or without accuracy aware tuning

Accuracy aware tuning is one of unique features provided by Intel® Neural Compressor quantization. This feature can be used to solve accuracy loss pain points brought by applying low precision quantization and other lossy optimization methods. Intel® Neural Compressor also supports to quantize all quantizable ops without accuracy tuning, user can decide whether to tune the model accuracy or not. Please check the [doc](https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md) for more details.

User can decide to tune the model accuracy by setting accuracy metric with goal in 'evaluator', and then setting 'evaluator' in Intel® Neural Compressor quantization pass. If not set, accuracy of the model will not be tuned.

```json
"evaluators": {
    "common_evaluator": {
        "metrics":[
            "name": "accuracy",
            "sub_types": [{"name": "accuracy_score", "priority": 1, "goal": {"type": "percent-max-degradation", "value": 0.02}}],
        ]
    }
},
"passes": {
    "quantization": {
        "type": "IncQuantization",
        "evaluator": "common_evaluator",
}
}

```

#### Example
First, install required packages according to passes.
```
python -m olive.workflows.run --config bert_inc_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config bert_inc_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_inc_config.json")
```

### Run with Intel® Neural Compressor static quantization

First, install required packages according to passes.
```
python -m olive.workflows.run --config bert_inc_static_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config bert_inc_static_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_inc_static_config.json")
```

> **Note**: Custom accuracy metric is used in `bert_inc_static_config.json`.

### Run with Intel® Neural Compressor dynamic quantization
First, install required packages according to passes.
```
python -m olive.workflows.run --config bert_inc_dynamic_config.json --setup
```
Then, optimize the model
```
python -m olive.workflows.run --config bert_inc_dynamic_config.json
```
or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("bert_inc_dynamic_config.json")
```