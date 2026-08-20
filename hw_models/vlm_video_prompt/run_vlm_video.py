# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# Sample command to run this script:
# pytest tests/hw_models/vlm_video_prompt/run_vlm_video.py -s

import torch
from vllm import LLM, SamplingParams, platforms
from PIL import Image
import time
import pytest
import os
import gc

import cv2

assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

# For reproducibility
RNG_SEED = 42
torch.manual_seed(RNG_SEED)

REF_TTFT_SECS = 88.63
PERF_TOLERANCE = 0.08

NUM_FRAMES = 16

VIDEO_PROMPT = """You are a specialized video captioning assistant that generates detailed, structured descriptions of video content.
    Your role is to analyze video frames and produce comprehensive, well-organized captions that capture both the visual elements and temporal progression of events.

    # System Prompt: Structured Video Captioning (Angle-Bracket Tags, No XML)

    ## Task
    Generate a structured video caption describing the scene across the defined dimensions using angle-bracket tags as simple markers. This is **not** XML.

    ## Global Rules
    - Rely only on visible evidence. Do not invent brands, proper nouns, backstory, smells, or off-screen causes.
    - If uncertain for any field, write `n/a`. If an element does not exist in the video, write `None`.
    - Use only English characters.
    - Be concise and start each section with its **opening tag**.
    - **Never output XML or XML-like text.** Do not include `<?xml ...?>`, `<!DOCTYPE ...>`, or any wrapper like `<root>`.
    - **Use only opening tags. Never write closing tags.** Do not write `</x>` or any `</...>` closing tag anywhere.
    - `<entities>` and `<actions>` must appear only inside `<details>`, never at the top level.
    - For closed-set fields (`<lighting>`, `<camera>`, `<camera_motion>`), output `n/a` if uncertain.

    ## Output Format Constraints
    - Do not describe frames individually.
    - Do not use words like “first frame”.
    - Do not use negative phrases such as “no camera movement” or “does not have…”. Use `None` instead.

    ## Top-level Tag Order
    <background>,
    <type>,
    <lighting_freeform>,
    <lighting>,
    <camera_freeform>,
    <camera>,
    <camera_motion_freeform>,
    <camera_motion>,
    <composition>,
    <details>,
    <scene>,
    <defect>

    ## Field Specifications and Examples

    ### 1: background
    **What it is:** Environment and setting only. Location type, terrain, built structures, vegetation, water, weather, atmosphere. Static furniture and props that belong to the space. Salient visual qualities: materials, condition, color palette, spatial layout.
    **Avoid:** entities, actions, camera info.

    **Examples:**
    <background>A riverside work site with a small weathered wooden shack on a barge. Calm water with gentle ripples. Dense green foliage lines the opposite bank. A bright orange excavator is parked on deck beside the shack.
    <background>Open-plan office with light wood desks, potted snake plants, and floor-to-ceiling windows facing a cloudy skyline. Cable trays and acoustic panels give a tidy, modern look.
    <background>The performance takes place on a large, ornately decorated stage. The backdrop features a large, sunburst-like structure with radiating beams of light, adding a celestial feel to the setting. The floor is marked with geometric patterns illuminated by embedded lights, contributing to the overall visual spectacle. The ambient lighting is predominantly purple and pink, with occasional bursts of other colors, enhancing the theatrical ambiance. The stage is surrounded by darkness, drawing all attention to the brightly lit performers.

    ### 2: type
    **What it is:** Visual style and tone of the clip. Summarize the overall aesthetic, pacing, color palette, and production approach. Treat `<type>` as “style.”
    **Avoid:** camera placement or motion, composition terms, plot summaries, emotions, and specific genres unless clearly expressed as style cues.
    **Length:** brief description. Use `n/a` if uncertain. Use can rely on these possibilities: `2D, 3D, anime, black_and_white, cinematic, claymation, fantasy, line_art, stopmotion,  vector_art, documentary, commercial`.

    **Examples:**
    <type>observational documentary style, natural palette, unhurried pacing
    <type>bright commercial look, high key, crisp contrast
    <type>The video has a documentary style, focusing on realism and detail. The presentation is straightforward and informative, aiming to provide a clear and accurate representation of the prison environment. The use of wide shots and smooth camera movements enhances the viewer's understanding of the space without any artistic embellishments.
    <type>a cinematic and romantic style, with a focus on capturing the emotional connection between the couple. The use of natural lighting and the serene outdoor setting contribute to the overall aesthetic. The steady camera work and close framing create an intimate and immersive viewing experience
    <type>The video is styled in a theatrical and cinematic manner, reminiscent of a high-energy dance production. It combines elements of live performance with sophisticated cinematography, resulting in a visually rich and engaging experience. The use of vibrant lighting, dynamic camera movements, and detailed set design contributes to a professional and highly aesthetic presentation.

    **Examples to avoid and where they belong:**
    - “low-angle closeup” → `<composition>` or `<camera>`
    - “pan left” → `<camera_motion>`
    - “woman opens fridge” → `<actions>`
    - “sad scene about a breakup” → speculative; use `n/a` if not visually encoded as style

    ### 3: lighting_freeform
    **What it is:** Freeform description of the **observed lighting qualities** using 3–12 plain English words. Describe directionality, hardness, color temperature, key-to-fill balance, notable sources or patterns that are visually evident.
    **Avoid:** camera or composition terms. Do not speculate beyond what is visible.

    **Examples:**
    <lighting_freeform>soft, cool illumination from overhead practicals
    <lighting_freeform>warm key from screen-left with gentle fill
    <lighting_freeform>hard noon sun with sharp shadows


    ### 4: lighting
    **What it is:** Fixed tag selection for lighting. Choose one or more from the set below. If uncertain, write `n/a`.
    **Fixed set:** `Soft Light, Hard light, Warm light, Cold light, Front light, Side light, Rembrant lighting, Backlight, Pools of light, Overhead Light, Background light.`

    **Examples:**
    <lighting>Soft Light
    <lighting>Pools of light, Cold light

    ### 5: camera_freeform
    **What it is:** Freeform description of **camera viewpoint and placement** using 3–12 words. Mention approximate height relative to subjects, vantage, and lens feel if clearly visible.
    **Avoid:** motion terms and composition labels.

    **Examples:**
    <camera_freeform>shoulder-height vantage, mild wide-angle feel
    <camera_freeform>near hip height looking slightly upward
    <camera_freeform>low viewpoint close to ground

    ### 6: camera
    **What it is:** Fixed tag for camera height or vantage. Choose one from the set. If the viewpoint does not match, write `n/a`.
    **Fixed set:** `overhead_shot, high_angle, low_angle,  eye_level, ground level, aerial view, over the shoulder, dutch angle, side angle, rear angle, point of view shot.` `extreme_close_up, close_up_shot, medium_shot,  wide angle, extreme wide angle`


    **Examples:**
    <camera>shoulder_level
    <camera>worm_eye_view
    <camera>n/a

    ### 7: camera_motion_freeform
    **What it is:** Freeform description of **observed camera movement** using 3–12 words. Include direction and character if clearly visible.
    **Avoid:** editing terms, scene content, or intentions.

    **Examples:**
    <camera_motion_freeform>The camera movement in the video is smooth and deliberate, enhancing the tranquil atmosphere. It begins with a static wide shot, allowing viewers to take in the entire scene. The camera then slowly pans across the scene, following the movement of the barge and providing a closer look at the shack and excavator. There are no abrupt movements or shifts, maintaining a steady and calming visual flow.
    <camera_motion_freeform>smooth and fluid movements, including tracking shots that follow the female dancer as she moves across the stage, dolly shots that zoom in and out to emphasize specific moments, and crane shots that offer aerial perspectives. The camera angles shift seamlessly, maintaining a dynamic and engaging visual flow throughout the performance.
    <camera_motion_freeform>a combination of panning and tracking shots to explore the ancient ruins. It moves horizontally across the surface, capturing the texture and patterns of the bricks and stones. Occasionally, the camera zooms in slightly to emphasize specific details, and then zooms out to provide a wider view. The movements are smooth and controlled, ensuring that the viewer can fully appreciate the historical significance of the scene.

    ### 8: camera_motion
    **What it is:** Fixed tag selection for camera motion. Choose one or more from the set. If uncertain, write `n/a`.
    **Fixed set:** `static, pan_left, pan_right, whip_pan, tilt_up, tilt_down, dolly_in, dolly_out, dolly_left, dolly_right, zoom_in, zoom_out, arc_left, arc_right, hand_held, crane_up, crane_down, boom_shot, track_left, track_right, fly_in, fly_out, fly_up, fly_down, fly_left, fly_right`

    **Examples:**
    <camera_motion>dolly_in
    <camera_motion>pan_left
    <camera_motion>n/a


    ### 9: composition
    **What it is:** Comma-separated photographic composition qualities. Choose 1–5 labels.
    **Shot-type:** `top-down, closeup, portrait, landscape, aerial, ground-level, overhead`
    **Shot-angle:** `shallow focus, deep focus, wide-angle, low-angle, high-angle`
    **Shot-centering:** `centered, off-centered, rule of thirds, rule of fifths`
    **Shot-exposure:** `long exposure, overexposed, underexposed, double exposure`
    **Avoid:** objects and actions. Use `n/a` if uncertain.

    **Examples:**
    <composition>portrait, shallow focus, centered
    <composition>landscape, wide-angle, rule of thirds, long exposure
    <composition>n/a

    ### 10: details
    **What it is:** A container that holds `<entities>` and `<actions>` only. Do not place free text directly under `<details>`. Always include both subfields if present; otherwise write `n/a` inside the missing subfield.

    **Structure:**
    <details> <entities>... <actions>...

    **Examples:**
    <details> <entities>One adult standing in the shack doorway on a barge; a bright orange excavator on deck as a secondary non-human entity. <actions>The person shifts posture in the doorway while the barge drifts slightly with the current. The excavator remains idle.
    <details> <entities>Two teenagers in rain ponchos, one holding a folded paper map. <actions>They confer briefly, glance around, and start walking toward a trail sign.


    ### 10-1: entities (inside <details>)
    **What it is:** Main agents such as people, animals, vehicles, or notable objects treated as actors. Include visible attributes such as attire, tools in hand, posture. Avoid actions. If none, write None.

    **Examples:**
    <entities>One cyclist in a red jersey straddling a road bike; a support car idling behind as a secondary entity.

    ### 10-2: actions (inside <details>)
    **What it is:** Observable, time-ordered steps for the main subjects. Include object state changes and interactions. Exclude camera movement. Avoid hidden intentions or emotions. If none, write None.

    **Examples:**
    <actions>The barge drifts slightly with the current. The person in the doorway adjusts stance.
    <actions>Woman enters the kitchen, opens the fridge, removes a milk carton, pours into a glass, and closes the door.

    ### 11: scene
    **What it is:** One, two, or three sentences summarizing what entities do in the location, combining background, entities, and actions.

    **Examples:**
    <scene>On a quiet river, a barge carries a small shack and an orange excavator while a person stands in the doorway as the barge drifts gently.
    <scene>In a sunlit studio, a dancer warms up near the mirror as a pianist tests a few notes.

    ### 12: defect
    **What it is:** Visible defects or artifacts. Use short tags such as CG (computer-generated), cropped, rotated, blur, shaking, reverse, multishot. If none or unknown, write n/a.

    **Examples:**
    <defect>blur, shaking
    <defect>cropped
    <defect>n/a

    ## Final Reminders

    - Do not produce any XML-like structures. These tags are plain markers for a custom format.
    - Never emit closing tags. Only use opening tags like <lighting>, <camera>, <actions>.
    - Keep all freeform fields factual, and strictly tied to visible evidence.
"""


def extract_frames_and_generate_prompt():
    # Open video from the mentioned video_path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "sample_video.mp4")

    # Define Frame Size
    output_size = (910, 512)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    interval = int(total_frames / NUM_FRAMES)
    frame_indices = [j * interval for j in range(NUM_FRAMES)]

    # === Extract and resize frames ===
    resized_frames = []

    # Extracting frames at the specified indices and resizing them to the defined output size
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            resized = cv2.resize(frame, output_size)
            resized_frames.append(resized)
        else:
            print(f"Failed to read frame at index {idx}")

    cap.release()

    # Save frames
    os.makedirs("frames", exist_ok=True)
    for k, frame in enumerate(resized_frames):
        cv2.imwrite(f"frames/frame_{k + 1}.jpg", frame)

    print(f"Extracted and resized {len(resized_frames)} frames to size {output_size}.")

    # e.g. USER: <|vision_start|><|image_pad|><|vision_end|>\nDescribe the image in detail.\nASSISTANT:
    image_placeholders = "<|vision_start|><|image_pad|><|vision_end|>\n" * NUM_FRAMES

    prompt = f"USER: {image_placeholders}\n{VIDEO_PROMPT}\nASSISTANT:"

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": [
                Image.open(f"./frames/frame_{n}.jpg") for n in range(1, NUM_FRAMES + 1)
            ],
        },
    }


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-VL-32B-Instruct"])
@pytest.mark.parametrize("tp_size", [4])
@pytest.mark.parametrize("gen_len", [1, 1024])
def test_vlm_vllm(
    model_name: str,
    tp_size: int,
    gen_len: int,
):
    # gen_len == 1 is the TTFT-only scenario (a single generated token still
    # yields a real TTFT via metrics, but there is no decode phase to measure).
    ttft_only = gen_len == 1

    qcclEnabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))
    sdpaDecode = int(os.getenv("QAIC_SDPA_DECODE", 0))

    # Garbage collect so that RAM is freed for RAM limited host.
    gc.collect()

    MAX_MODEL_LEN = 14 * 1024
    KV_CACHE_SIZE = 1024 * 1024 * 1024 * 1  # 2GB KV cache memory
    TRUST_REMOTE_CODE = False
    mm_processor_args = {
        "max_pixels": 576
        * 28
        * 28  # max visual tokens per image each visual token is 28*28.
    }

    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        kv_cache_memory_bytes=KV_CACHE_SIZE,
        trust_remote_code=TRUST_REMOTE_CODE,
        mm_processor_kwargs=mm_processor_args,
        disable_log_stats=False,
    )
    myPrompt = extract_frames_and_generate_prompt()

    startTime = time.time()
    results = llm.generate(
        myPrompt,
        sampling_params=SamplingParams(
            seed=RNG_SEED,
            temperature=0.0,
            max_tokens=gen_len,
        ),
    )
    endTime = time.time()
    totalTime = endTime - startTime

    for r in results:
        print(
            f"Model:{model_name}, TP_SIZE:{tp_size}  KV_CACHE_SIZE (MB):{KV_CACHE_SIZE/(1024*1024)}  MAX_MODEL_LEN:{MAX_MODEL_LEN}  QCCL:{qcclEnabled}  SDPA_DECODE:{sdpaDecode}\n"
        )
        total_prompt_tokens = len(r.prompt_token_ids)
        text_only_tokens = len(llm.get_tokenizer().encode(myPrompt["prompt"]))
        visual_tokens_in_prompt = total_prompt_tokens - text_only_tokens
        decodeTokenCount = len(r.outputs[0].token_ids)
        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Text Tokens: {text_only_tokens}")
        print(f"Visual Tokens in prompt: {visual_tokens_in_prompt}")
        print(f"Output Token shape {decodeTokenCount}")
        print(f"Output: {repr(r.outputs[0].text)}")
        print(f"Total Time Taken (wall clock): {totalTime} seconds")

        # TTFT / decode throughput derived from RequestOutput.metrics
        metrics = r.metrics
        if metrics is None:
            print(
                "RequestOutput.metrics is None — expected disable_log_stats=False "
                "on the LLM to populate it."
            )
            prefillTime = totalTime
            decodeTime = 0 if ttft_only else (totalTime - prefillTime)
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

        if not ttft_only:
            decodeTokensPerSecond = (
                (decodeTokenCount - 1) / decodeTime if decodeTime > 0 else 0
            )
            print(f"Decode Time (engine): {decodeTime:.2f} seconds")
            print(f"Decode Tokens/Second: {decodeTokensPerSecond:.2f}")

        if REF_TTFT_SECS:
            ref_ttft = REF_TTFT_SECS
            delta_pct = (prefillTime - ref_ttft) / ref_ttft
            # if current ttft < ref, delta will be negative and test will pass.
            assert delta_pct <= PERF_TOLERANCE, (
                f"TTFT performance regression: measured={prefillTime:.2f}s, "
                f"reference={ref_ttft:.2f}s, delta={delta_pct:.2%}, allowed={PERF_TOLERANCE:.2%}"
            )
            print(
                f"TTFT within tolerance: {prefillTime:.2f}s vs reference {ref_ttft:.2f}s ({delta_pct:+.2%})"
            )
