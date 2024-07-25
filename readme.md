# Doubutsu Image Describer for ComfyUI

This custom node for ComfyUI allows you to use the Doubutsu small VLM model to describe images.
Credit and further information on Doubutsu: https://huggingface.co/qresearch/doubutsu-2b-pt-756

![image](https://github.com/user-attachments/assets/766be6c5-b1f0-4a2e-b98f-8fda661051b9)


## Installation

1. Clone this repository into your ComfyUI's `custom_nodes` directory:
git clone https://github.com/EnragedAntelope/comfyui-doubutsu-describer.git
2. Install the required dependencies:
pip install -r requirements.txt
3. Download the model files:
- Create a `models` directory in the root of this repository (ComfyUI\custom_nodes\ComfyUI-Doubutsu-Describer).
- Download the model files for "qresearch/doubutsu-2b-pt-756" from Hugging Face and place them in `models/qresearch/doubutsu-2b-pt-756/`.
- Download the adapter files for "qresearch/doubutsu-2b-lora-756-docci" and place them in `models/qresearch/doubutsu-2b-lora-756-docci/`.

You can download these files manually from the Hugging Face website or use the Hugging Face CLI:

  Open a command prompt, navigate to your ComfyUI\custom_nodes\ComfyUI-Doubutsu-Describer directory, then execute:

  'huggingface-cli download qresearch/doubutsu-2b-pt-756 --local-dir models/qresearch/doubutsu-2b-pt-756'

  'huggingface-cli download qresearch/doubutsu-2b-lora-756-docci --local-dir models/qresearch/doubutsu-2b-lora-756-docci'

4. Restart ComfyUI

## Usage

After installation, you'll find a new node called "Doubutsu Image Describer" in the "image/text" category. Connect an image to its input, and it will generate a description based on the provided question.

## Parameters

- `image`: The input image to describe
- `question`: The question to ask about the image (default: "Describe the image")
- `max_new_tokens`: Maximum number of tokens to generate (default: 128)
- `temperature`: Controls randomness in generation (default: 0.1)
- `precision`: Choose between float16 or bfloat16 for inference. If your GPU supports it, bfloat16 should be quicker.

## License

[Apache 2.0]
