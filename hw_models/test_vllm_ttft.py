#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

import torch
from vllm import LLM, SamplingParams, platforms
from transformers import AutoTokenizer
import time
import pytest
import os
import gc

assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

# For reproducibility
RNG_SEED = 42
torch.manual_seed(RNG_SEED)


def get_input_prompt(prompt_length=128, tokenizer=None):
    # Creating specific length prompt

    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")

    prompt = (
        "Artificial intelligence (AI) has rapidly evolved over the past few decades, transforming from a niche area of computer science into a foundational technology that permeates nearly every aspect of modern life. "
        "The journey of AI began with the ambition to create machines that could mimic human intelligence, reason, and learn. Early pioneers in the field, such as Alan Turing and John McCarthy, laid the groundwork for what would become a revolution in computing. "
        "Turings famous question, “Can machines think?” sparked philosophical debates and technical challenges that continue to this day. The Turing Test, proposed as a measure of a machines ability to exhibit intelligent behavior indistinguishable from that of a human, remains a touchstone in discussions about AIs capabilities and limitations. "
        "In the early years, AI research focused on symbolic reasoning and rule-based systems. These approaches sought to encode human knowledge and logic into computer programs, enabling machines to solve problems, play games like chess, and perform basic language processing. "
        "However, these systems were limited by their reliance on explicit rules and struggled to handle the complexity and ambiguity inherent in real-world environments. The so-called “AI winter” periods, marked by reduced funding and optimism, reflected the challenges of scaling early AI systems to practical applications. "
        "The resurgence of AI in the late 20th and early 21st centuries was fueled by advances in machine learning, particularly the development of artificial neural networks. Inspired by the structure and function of the human brain, neural networks enabled computers to learn from data rather than relying solely on hand-crafted rules. "
        "The advent of deep learning, characterized by multi-layered neural networks capable of extracting hierarchical features from raw data, revolutionized fields such as computer vision, natural language processing, and speech recognition. "
        "Landmark achievements, such as the defeat of world champions in games like Go and poker, showcased the power of deep reinforcement learning and the ability of AI systems to master complex, strategic tasks. "
        "One of the most significant drivers of AIs recent progress has been the exponential growth of data and computational power. The proliferation of digital devices, sensors, and the internet has generated vast amounts of data, providing the raw material for training sophisticated AI models. "
        "At the same time, advances in hardware, including graphics processing units (GPUs) and specialized AI accelerators, have enabled researchers to train larger and more complex models than ever before. "
        "Cloud computing platforms have democratized access to high-performance computing resources, allowing organizations of all sizes to experiment with and deploy AI solutions. "
        "AIs impact on society is profound and multifaceted. In healthcare, AI-powered systems assist in diagnosing diseases, predicting patient outcomes, and personalizing treatment plans. "
        "Radiology, pathology, and genomics are among the fields where AI algorithms analyze medical images and genetic data with remarkable accuracy, often surpassing human experts. "
        "In finance, AI-driven trading algorithms, fraud detection systems, and customer service chatbots have transformed the industry, enabling faster and more efficient operations. "
        "Autonomous vehicles, powered by AI, promise to revolutionize transportation by reducing accidents, improving traffic flow, and increasing mobility for individuals unable to drive. "
        "Natural language processing (NLP) has seen remarkable progress, with AI models now capable of understanding, generating, and translating human language with unprecedented fluency. "
        "Virtual assistants, such as Siri, Alexa, and Google Assistant, leverage NLP to interact with users, answer questions, and perform tasks. "
        "Large language models, like GPT-3 and its successors, can generate coherent and contextually relevant text, write code, compose poetry, and even engage in philosophical debates. "
        "These capabilities have opened new possibilities in education, content creation, and customer service, while also raising concerns about misinformation, bias, and the ethical use of AI-generated content. "
        "The integration of AI into everyday devices and services has led to the emergence of smart homes, personalized recommendations, and intelligent automation. "
        "Smart thermostats, security systems, and appliances learn user preferences and optimize energy consumption, enhancing comfort and efficiency. "
        "Streaming platforms and e-commerce sites use AI algorithms to recommend movies, music, and products tailored to individual tastes, driving engagement and sales. "
        "In manufacturing and logistics, AI-powered robots and predictive maintenance systems increase productivity, reduce downtime, and improve supply chain resilience. "
        "Despite its many benefits, the widespread adoption of AI also presents significant challenges and risks. Concerns about privacy, security, and the potential for job displacement have prompted calls for responsible AI development and governance. "
        "The use of AI in surveillance, facial recognition, and social scoring has sparked debates about civil liberties and the appropriate limits of technology. "
        "Algorithmic bias, stemming from biased training data or flawed model design, can perpetuate and amplify social inequalities, leading to unfair outcomes in areas such as hiring, lending, and law enforcement. "
        "Ensuring transparency, accountability, and fairness in AI systems is a critical priority for researchers, policymakers, and industry leaders. "
        "The future of AI holds both promise and uncertainty. As AI systems become more capable and autonomous, questions about control, alignment with human values, and the potential for unintended consequences become increasingly salient. "
        "The development of explainable AI, which seeks to make the decision-making processes of AI systems more transparent and understandable, is an active area of research. "
        "Collaborative efforts between academia, industry, and government are essential to establish ethical guidelines, standards, and regulatory frameworks that balance innovation with societal well-being. "
        "Education and workforce development are also key to harnessing the benefits of AI while mitigating its risks. Preparing individuals for the jobs of the future requires a focus on digital literacy, critical thinking, and lifelong learning. "
        "Interdisciplinary collaboration, combining expertise in computer science, ethics, law, and social sciences, is necessary to address the complex challenges posed by AI. "
        "Public engagement and dialogue are vital to ensure that AI technologies reflect diverse perspectives and serve the broader interests of society. "
        "In conclusion, artificial intelligence represents one of the most transformative technologies of our time. Its rapid evolution, driven by advances in machine learning, data, and computing power, has enabled breakthroughs across a wide range of domains. "
        "AIs potential to improve healthcare, education, transportation, and many other fields is immense, but realizing this potential requires careful attention to ethical, social, and economic considerations. "
        "By fostering responsible innovation, promoting transparency and fairness, and investing in education and collaboration, we can shape the future of AI to benefit individuals and society as a whole. The journey of AI is far from over, and its ultimate impact will depend on the choices we make today."
        "Artificial intelligence (AI) has rapidly evolved over the past few decades, transforming from a niche area of computer science into a foundational technology that permeates nearly every aspect of modern life. "
        "The journey of AI began with the ambition to create machines that could mimic human intelligence, reason, and learn. Early pioneers in the field, such as Alan Turing and John McCarthy, laid the groundwork for what would become a revolution in computing. "
        "Turings famous question, “Can machines think?” sparked philosophical debates and technical challenges that continue to this day. The Turing Test, proposed as a measure of a machines ability to exhibit intelligent behavior indistinguishable from that of a human, remains a touchstone in discussions about AIs capabilities and limitations. "
        "In the early years, AI research focused on symbolic reasoning and rule-based systems. These approaches sought to encode human knowledge and logic into computer programs, enabling machines to solve problems, play games like chess, and perform basic language processing. "
        "However, these systems were limited by their reliance on explicit rules and struggled to handle the complexity and ambiguity inherent in real-world environments. The so-called “AI winter” periods, marked by reduced funding and optimism, reflected the challenges of scaling early AI systems to practical applications. "
        "The resurgence of AI in the late 20th and early 21st centuries was fueled by advances in machine learning, particularly the development of artificial neural networks. Inspired by the structure and function of the human brain, neural networks enabled computers to learn from data rather than relying solely on hand-crafted rules. "
        "The advent of deep learning, characterized by multi-layered neural networks capable of extracting hierarchical features from raw data, revolutionized fields such as computer vision, natural language processing, and speech recognition. "
        "Landmark achievements, such as the defeat of world champions in games like Go and poker, showcased the power of deep reinforcement learning and the ability of AI systems to master complex, strategic tasks. "
        "One of the most significant drivers of AIs recent progress has been the exponential growth of data and computational power. The proliferation of digital devices, sensors, and the internet has generated vast amounts of data, providing the raw material for training sophisticated AI models. "
        "At the same time, advances in hardware, including graphics processing units (GPUs) and specialized AI accelerators, have enabled researchers to train larger and more complex models than ever before. "
        "Cloud computing platforms have democratized access to high-performance computing resources, allowing organizations of all sizes to experiment with and deploy AI solutions. "
        "AIs impact on society is profound and multifaceted. In healthcare, AI-powered systems assist in diagnosing diseases, predicting patient outcomes, and personalizing treatment plans. "
        "Radiology, pathology, and genomics are among the fields where AI algorithms analyze medical images and genetic data with remarkable accuracy, often surpassing human experts. "
        "In finance, AI-driven trading algorithms, fraud detection systems, and customer service chatbots have transformed the industry, enabling faster and more efficient operations. "
        "Autonomous vehicles, powered by AI, promise to revolutionize transportation by reducing accidents, improving traffic flow, and increasing mobility for individuals unable to drive. "
        "Natural language processing (NLP) has seen remarkable progress, with AI models now capable of understanding, generating, and translating human language with unprecedented fluency. "
        "Virtual assistants, such as Siri, Alexa, and Google Assistant, leverage NLP to interact with users, answer questions, and perform tasks. "
        "Large language models, like GPT-3 and its successors, can generate coherent and contextually relevant text, write code, compose poetry, and even engage in philosophical debates. "
        "These capabilities have opened new possibilities in education, content creation, and customer service, while also raising concerns about misinformation, bias, and the ethical use of AI-generated content. "
        "The integration of AI into everyday devices and services has led to the emergence of smart homes, personalized recommendations, and intelligent automation. "
        "Smart thermostats, security systems, and appliances learn user preferences and optimize energy consumption, enhancing comfort and efficiency. "
        "Streaming platforms and e-commerce sites use AI algorithms to recommend movies, music, and products tailored to individual tastes, driving engagement and sales. "
        "In manufacturing and logistics, AI-powered robots and predictive maintenance systems increase productivity, reduce downtime, and improve supply chain resilience. "
        "Despite its many benefits, the widespread adoption of AI also presents significant challenges and risks. Concerns about privacy, security, and the potential for job displacement have prompted calls for responsible AI development and governance. "
        "The use of AI in surveillance, facial recognition, and social scoring has sparked debates about civil liberties and the appropriate limits of technology. "
        "Algorithmic bias, stemming from biased training data or flawed model design, can perpetuate and amplify social inequalities, leading to unfair outcomes in areas such as hiring, lending, and law enforcement. "
        "Ensuring transparency, accountability, and fairness in AI systems is a critical priority for researchers, policymakers, and industry leaders. "
        "The future of AI holds both promise and uncertainty. As AI systems become more capable and autonomous, questions about control, alignment with human values, and the potential for unintended consequences become increasingly salient. "
        "The development of explainable AI, which seeks to make the decision-making processes of AI systems more transparent and understandable, is an active area of research. "
        "Collaborative efforts between academia, industry, and government are essential to establish ethical guidelines, standards, and regulatory frameworks that balance innovation with societal well-being. "
        "Education and workforce development are also key to harnessing the benefits of AI while mitigating its risks. Preparing individuals for the jobs of the future requires a focus on digital literacy, critical thinking, and lifelong learning. "
        "Interdisciplinary collaboration, combining expertise in computer science, ethics, law, and social sciences, is necessary to address the complex challenges posed by AI. "
        "Public engagement and dialogue are vital to ensure that AI technologies reflect diverse perspectives and serve the broader interests of society. "
        "In conclusion, artificial intelligence represents one of the most transformative technologies of our time. Its rapid evolution, driven by advances in machine learning, data, and computing power, has enabled breakthroughs across a wide range of domains. "
        "AIs potential to improve healthcare, education, transportation, and many other fields is immense, but realizing this potential requires careful attention to ethical, social, and economic considerations. "
        "By fostering responsible innovation, promoting transparency and fairness, and investing in education and collaboration, we can shape the future of AI to benefit individuals and society as a whole. The journey of AI is far from over, and its ultimate impact will depend on the choices we make today."
    )

    tokens = tokenizer.encode(prompt)

    # Check the number of tokens
    num_tokens = len(tokens)

    while num_tokens < prompt_length:
        # If the prompt is shorter than prompt_length, repeat the prompt until it reaches the desired length
        tokens += tokens
        num_tokens = len(tokens)

    # Ensure the prompt is exactly prompt_length tokens
    tokens = tokens[:prompt_length]

    # Decode the tokens back to text
    prompt_sliced = tokenizer.decode(tokens, skip_special_tokens=True)
    return prompt_sliced


# Performance references keyed by (model_name, tp_size, prompt_length).
# "ttft" is the maximum allowed wall-clock TTFT in seconds.
PERF_TOLERANCE = 0.08
REF_PERF = {
    ("meta-llama/Llama-3.1-8B-Instruct", 4, 8192): {"ttft": 7.96},
    ("meta-llama/Llama-3.1-8B-Instruct", 4, 16384): {"ttft": 23.21},
    ("unsloth/gpt-oss-20b-BF16", 4, 8192): {"ttft": 155.78},
    ("openai/gpt-oss-20b", 4, 8192): {"ttft": 156.04},
}


@pytest.fixture(scope="function")
def env_sdpa_decode(request):
    flag = str(request.param)
    os.environ["QAIC_SDPA_DECODE"] = flag
    yield
    os.environ.pop("QAIC_SDPA_DECODE", None)


MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "unsloth/gpt-oss-20b-BF16",
    "openai/gpt-oss-20b",
]


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("tp_size", [4, 8, 16])
@pytest.mark.parametrize("gen_length", [1, 100, 1024, 10000, 20000])
@pytest.mark.parametrize(
    "max_num_batched_tokens", [8192, 16384, 32000, 64000, 100000, 127000, 128000]
)
@pytest.mark.parametrize("prompt_length", [8192, 16384, 32000, 64000, 100000, 127000, 128000])
@pytest.mark.parametrize("env_sdpa_decode", ["1", "0"], indirect=True)
def test_vllm_ttft(
    env_sdpa_decode,
    model_name: str,
    tp_size: int,
    prompt_length: int,
    gen_length: int,
    max_num_batched_tokens: int,
):
    qcclEnabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))
    sdpaDecode = int(os.getenv("QAIC_SDPA_DECODE", 0))

    # Garbage collect so that RAM is freed for RAM limited host.
    gc.collect()

    MAX_MODEL_LEN = min(128000, prompt_length + gen_length)
    KV_CACHE_SIZE = 1024 * 1024 * 1024 * 2  # 2GB KV cache memory
    TRUST_REMOTE_CODE = False

    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        kv_cache_memory_bytes=KV_CACHE_SIZE,
        trust_remote_code=TRUST_REMOTE_CODE,
        disable_log_stats=False,
        max_num_batched_tokens=max_num_batched_tokens,
        # hf_overrides={"num_hidden_layers": 2},
        # load_format="dummy",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = get_input_prompt(prompt_length=prompt_length, tokenizer=tokenizer)

    startTime = time.perf_counter()
    results = llm.generate(
        prompt,
        sampling_params=SamplingParams(
            seed=RNG_SEED,
            temperature=0.0,
            max_tokens=gen_length,
        ),
    )
    endTime = time.perf_counter()
    totalGenerateTime = endTime - startTime

    for r in results:
        print(
            f"Model:{model_name}, TP_SIZE:{tp_size}  KV_CACHE_SIZE (MB):{KV_CACHE_SIZE/(1024*1024)}  MAX_MODEL_LEN:{MAX_MODEL_LEN}  QCCL:{qcclEnabled}  SDPA_DECODE:{sdpaDecode}\n"
        )
        print(f"Prompt Token Id Shape: {len(r.prompt_token_ids)}")
        decodeTokens = len(r.outputs[0].token_ids)
        print(f"Output Token shape {decodeTokens}, Output: {repr(r.outputs[0].text)}")
        print(f"Total Generation Time: {totalGenerateTime:.3f} seconds")

        # Decode throughput derived from RequestOutput.metrics
        metrics = r.metrics
        if metrics is None:
            print(
                "RequestOutput.metrics is None — expected disable_log_stats=False "
                "on the LLM to populate it."
            )
        else:
            # first_token_latency is the true end-user TTFT (wall clock,
            # arrival -> first token). prefill_time (engine-core monotonic
            # clock, scheduled -> first token) excludes host-side CPU
            # preprocessing (tokenization) and queueing, so rendering_time
            # recovers that host-side portion by subtracting the two (both
            # share the same "first token produced" instant, cancelling out).
            metricsTtft = metrics.first_token_latency
            prefillTime = metrics.first_token_ts - metrics.scheduled_ts
            renderingTime = metricsTtft - prefillTime
            print(f"TTFT (arrival to first token): {metricsTtft:.2f} seconds")
            print(
                f"Rendering Prompts Time (host-side preprocessing): {renderingTime:.2f} seconds"
            )
            print(f"Prefill Time (scheduled to first token): {prefillTime:.2f} seconds")

            decodeTime = metrics.last_token_ts - metrics.first_token_ts
            decodeTokensPerSecond = (
                (decodeTokens - 1) / decodeTime if decodeTime > 0 else 0
            )
            print(f"Decode Time (engine): {decodeTime:.2f} seconds")
            print(f"Decode Tokens/Second: {decodeTokensPerSecond:.2f}")

    perf_key = (model_name, tp_size, prompt_length)
    ref_perf = REF_PERF.get(perf_key)
    if ref_perf is None:
        print(
            f"Missing performance reference for config (model_name, tp_size, prompt_length): {perf_key}"
        )
    else:
        ref_ttft = ref_perf["ttft"]
        delta_pct = (prefillTime - ref_ttft) / ref_ttft
        assert delta_pct <= PERF_TOLERANCE, (
            f"TTFT performance regression: measured={prefillTime:.2f}s, "
            f"reference={ref_ttft:.2f}s, delta={delta_pct:.2%}, allowed={PERF_TOLERANCE:.2%}"
        )
        print(
            f"TTFT within tolerance: {prefillTime:.2f}s vs reference {ref_ttft:.2f}s ({delta_pct:+.2%})"
        )
