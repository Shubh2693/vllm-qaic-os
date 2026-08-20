# Adapted from vllm/examples/offline_inference/basic/generate.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm import LLM, EngineArgs, platforms
from vllm.utils.argparse_utils import FlexibleArgumentParser

# verify vllm is indeed running on qaic
assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    num_devices = torch.qaic.device_count()
    parser.set_defaults(
        model="meta-llama/Llama-3.1-8B",
        tensor_parallel_size=num_devices,
        max_model_len=2048,
        enforce_eager=True,
    )

    return parser


def main(args: dict):
    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object and make it deterministic
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = 0.0
    sampling_params.seed = 42
    sampling_params.top_p = 1.0
    sampling_params.top_k = -1

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Assert expected outputs.
    expected_outputs = [
        " and I'm writing you today to learn more about the 2019 Ford F",
        " the head of state and head of government of the United States, indirectly elected to",
        " a city of many faces. It is a city of history, culture, and",
        " here, and it’s already changing the way we live and work. From self",
    ]
    for output, expected_output in zip(outputs, expected_outputs):
        assert output.outputs[0].text == expected_output


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
