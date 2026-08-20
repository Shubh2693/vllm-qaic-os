#########################################################################################
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
#########################################################################################

import torch
from vllm import LLM, SamplingParams, platforms
from vllm.config import CompilationConfig, CompilationMode
from PIL import Image
import time
import pytest
import os
import gc
import io
import re
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tabulate import tabulate

assert (
    platforms.current_platform.device_type == "qaic"
), "vLLM could not detect qaic plugin"

# Fixed seed is used so that text generation stays deterministic across runs,
# making both accuracy and performance comparisons stable.
RNG_SEED = 42
torch.manual_seed(RNG_SEED)

# Prompt templates keyed by prompt_type, used in perf and accuracy tests.
TEXT_PROMPT = {
    "describe": "Describe the image in detail.",
    "video_prompt": """You are a specialized video captioning assistant that generates detailed, structured descriptions of video content.
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
                    """,
}


@pytest.fixture(scope="function")
def env_sdpa_decode(request):
    # This fixture sets QAIC_SDPA_DECODE per test parameter and restores the
    # environment afterwards so each parametrized test runs in isolation.
    flag = str(request.param)
    os.environ["QAIC_SDPA_DECODE"] = flag
    yield
    os.environ.pop("QAIC_SDPA_DECODE", None)


# Models covered by this regression test. The test validates both generated
# text accuracy and a small set of performance metrics against stored reference
# values for each supported runtime configuration.
VLM_MODELS = [
    # Qwen 2.5 VL
    "Qwen/Qwen2.5-VL-3B-Instruct",  # TP: 2,4
    "Qwen/Qwen2.5-VL-7B-Instruct",  # TP: 2,4
    "Qwen/Qwen2.5-VL-32B-Instruct",  # TP: 2,4,8
    # Qwen 3 VL
    "Qwen/Qwen3-VL-32B-Instruct",  # TP: 2,4,8,16
    "Qwen/Qwen3-VL-30B-A3B-Instruct",  # MoE; TP: 4
    # InternVL
    "OpenGVLab/InternVL3_5-8B-Instruct",  # TP: 2,4
    "OpenGVLab/InternVL2_5-38B",  # TP: 2,4,8
]

# Accuracy references.
# Key format:
#   (model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode)
# These are exact-match text references for deterministic decoding runs.
REF_OUTPUT = {
    (
        "OpenGVLab/InternVL2_5-38B",
        4,
        True,
        1,
        0,
    ): ''' The image features two Qualcomm Cloud AI 100 products. On the left, there is a silver-colored card with a blue design element and the text "Qualcomm Cloud AI 100" printed on it. On the right, there is a black card with the text "Qualcomm Cloud AI 100 Ultra" and a series of connectors visible on the side. The background is white, and the text above the products reads "Qualcomm Cloud AI 100." Below the products, there is a description stating, "Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "OpenGVLab/InternVL2_5-38B",
        4,
        True,
        1,
        1,
    ): """ The image features two Qualcomm Cloud AI 100 products, showcasing their design and branding. \n\nOn the left, there is a silver-colored card with a sleek, modern design. It has a prominent blue and silver logo with the text "Qualcomm Cloud AI 100" printed on it. The card has a gold-colored edge connector, indicating it is likely a PCIe card.\n\nOn the right, there is a black card with a similar design, also labeled "Qualcomm Cloud AI 100 Ultra." This card has a green circuit board visible along the edge, and it also has a gold-colored edge connector.\n\n""",
    (
        "OpenGVLab/InternVL2_5-38B",
        4,
        False,
        1,
        1,
    ): ''' The image features two Qualcomm Cloud AI 100 products. On the left, there is a silver-colored card with a blue design element and the text "Qualcomm Cloud AI 100" printed on it. On the right, there is a black card with the text "Qualcomm Cloud AI 100 Ultra" and a series of connectors visible on the side. The background is white, and the text above the products reads "Qualcomm Cloud AI 100." Below the products, there is a description stating, "Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "OpenGVLab/InternVL2_5-38B",
        4,
        False,
        1,
        0,
    ): ''' The image features two Qualcomm Cloud AI 100 products. On the left, there is a silver-colored card with a blue design element and the text "Qualcomm Cloud AI 100" printed on it. On the right, there is a black card with the text "Qualcomm Cloud AI 100 Ultra" and a series of connectors visible on the side. The background is white, and the text above the products reads "Qualcomm Cloud AI 100." Below the products, there is a description stating, "Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "Qwen/Qwen2.5-VL-32B-Instruct",
        4,
        True,
        1,
        0,
    ): ''' The image showcases two products from Qualcomm, specifically the Qualcomm Cloud AI 100 series. Here\'s a detailed description:\n\n### **Text Elements:**\n1. **Title:**\n   - At the top of the image, the text reads: **"Qualcomm® Cloud AI 100"** in bold, black font. The "Qualcomm" brand name is accompanied by the registered trademark symbol (®).\n   \n2. **Description:**\n   - Below the title, there is a brief description in black text that states: **"Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "Qwen/Qwen2.5-VL-32B-Instruct",
        4,
        True,
        1,
        1,
    ): """ The image showcases two PCIe cards, labeled as the "Qualcomm® Cloud AI 100." The text on the image highlights that these cards are "Purpose-built for high performance, low-power AI processing in the cloud and edge." Here\'s a detailed description:\n\n### **Left Card:**\n1. **Design:**\n   - The card has a sleek, modern design with a metallic finish.\n   - The surface is primarily silver with a dark blue stripe running horizontally along the top edge.\n   - The blue stripe features the Qualcomm logo, which is a stylized "Q" in white, accompanied by the text "Qualcomm""",
    (
        "Qwen/Qwen2.5-VL-32B-Instruct",
        4,
        False,
        1,
        0,
    ): ''' The image showcases two products from Qualcomm, specifically the Qualcomm Cloud AI 100 series. Here\'s a detailed description:\n\n### **Text Elements:**\n1. **Title:**\n   - At the top of the image, the text reads: **"Qualcomm® Cloud AI 100"** in bold, black font. The "Qualcomm" brand name is accompanied by the registered trademark symbol (®).\n   \n2. **Description:**\n   - Below the title, there is a brief description in black text that states: **"Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "Qwen/Qwen2.5-VL-32B-Instruct",
        4,
        False,
        1,
        1,
    ): ''' The image showcases two products from Qualcomm, specifically the Qualcomm Cloud AI 100 series. Here\'s a detailed description:\n\n### **Text Elements:**\n1. **Title:**\n   - At the top of the image, the text reads: **"Qualcomm® Cloud AI 100"** in bold, black font. The "Qualcomm" brand name is accompanied by the registered trademark symbol (®).\n   \n2. **Description:**\n   - Below the title, there is a brief description in black text that states: **"Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        1,
        True,
        1,
        1,
    ): """ The image is a promotional graphic for the Qualcomm® Cloud AI 100, a purpose-built device designed for high-performance, low-power AI processing in both the cloud and edge environments. The graphic features two main components of the Qualcomm® Cloud AI 100:\n\n1. **Card with Blue and Black Design**: This component is likely a PCIe card, which is a type of expansion card used to add functionality to a computer. The card has a sleek, modern design with a blue and black color scheme. The text on the card reads "Qualcomm® Cloud AI 100," indicating its brand and model.\n\n2""",
    (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        4,
        True,
        1,
        1,
    ): """ The image is a promotional graphic for the Qualcomm® Cloud AI 100, a purpose-built device designed for high-performance, low-power AI processing in both the cloud and edge environments. The graphic features two main components of the Qualcomm® Cloud AI 100:\n\n1. **Card with Blue and Black Design**: This component is likely a PCIe card, which is a type of add-in card used to expand the functionality of a computer. The card has a sleek, modern design with a blue and black color scheme. The text on the card reads "Qualcomm® Cloud AI 100," indicating its brand and purpose""",
    (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        4,
        True,
        1,
        1,
    ): """ The image showcases a product advertisement for the "Qualcomm Cloud AI 100," which is a specialized hardware component designed for high-performance, low-power AI processing. The advertisement highlights its suitability for use in both cloud and edge computing environments. The image features two views of the hardware: one is a side view of the card, and the other is a front view. The side view displays the card\'s profile, showing its size and the connectors on the back, which are likely for connecting to a motherboard or server. The front view provides a closer look at the card\'s surface, which is labeled with the "Qualcomm Cloud AI""",
    (
        "Qwen/Qwen3-VL-32B-Instruct",
        4,
        True,
        1,
        1,
    ): """ The image is a promotional graphic for the **Qualcomm Cloud AI 100**, showcasing two physical hardware modules designed for AI processing. The layout is clean and professional, with a white background and black text.\n\n---\n\n### **Top Section: Title**\nAt the top center, in a large, bold, sans-serif font, is the product name:\n> **Qualcomm® Cloud AI 100**\n\nThe "®" symbol indicates a registered trademark.\n\n---\n\n### **Left Side: Hardware Module (Angle View)**\nOn the left, there is a 3D angled view of a rectangular, silver-colored hardware module with a""",
    (
        "OpenGVLab/InternVL3_5-8B-Instruct",
        4,
        True,
        1,
        1,
    ): """ The image is a promotional graphic for Qualcomm\'s Cloud AI 100. It features two views of the hardware component: a side view and a front view. The side view shows the component with a metallic finish and the Qualcomm logo, while the front view displays the component with a black finish and the text "Qualcomm Cloud AI 100 Ultra" along with the Qualcomm logo. The text in the image reads: "Qualcomm® Cloud AI 100. Purpose-built for high performance, low-power AI processing in the cloud and edge." The overall design is sleek and professional, emphasizing the product\'s capabilities in AI""",
    (
        "Qwen/Qwen3-VL-32B-Instruct",
        4,
        False,
        1,
        1,
    ): """ The image is a promotional graphic for the **Qualcomm Cloud AI 100**, a specialized AI accelerator designed for high-performance, low-power AI processing in cloud and edge environments.\n\n---\n\n### **Visual Layout:**\n\nThe image is horizontally oriented and divided into three main sections:\n\n1. **Top Center – Branding and Product Name:**\n   - The text **"Qualcomm® Cloud AI 100"** is prominently displayed in a clean, modern sans-serif font.\n   - The "Qualcomm" logo includes the registered trademark symbol (®).\n   - The text is black and centered at the top of the image""",
    (
        "Qwen/Qwen2.5-VL-32B-Instruct",
        4,
        False,
        0,
        1,
    ): ''' The image showcases two products from Qualcomm, specifically the Qualcomm Cloud AI 100 series. Here\'s a detailed description:\n\n### **Text Elements:**\n1. **Title:**\n   - At the top of the image, the text reads: **"Qualcomm® Cloud AI 100"** in bold, black font. The "Qualcomm" brand name is accompanied by the registered trademark symbol (®).\n   \n2. **Description:**\n   - Below the title, there is a brief description in black text that states: **"Purpose-built for high performance, low-power AI processing in the cloud and edge."''',
    (
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        4,
        False,
        1,
        1,
    ): """This image is a promotional graphic for the Qualcomm Cloud AI 100, a hardware product designed for artificial intelligence processing. The image is split into two main sections, showcasing the product from different angles and providing descriptive text.\n\nOn the left side of the image, a silver-colored, low-profile expansion card is shown. This card features a blue accent stripe along its side and has the "Qualcomm" logo printed on its surface. It is designed to be installed into a computer system, likely a server, via a PCIe slot.\n\nOn the right side, a black, full-height expansion card is displayed. This card is labeled "Qual""",
}
# Accuracy references for torch.compile mode (enforce_eager=False).
# Keyed separately from REF_OUTPUT because torch.compile with the inductor
# backend may produce different floating-point results than eager execution,
# leading to different token outputs even with temperature=0.0.
# Key format: (model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode)
# NOTE: Populate these values by running test_vlm_vllm_compile and capturing
# the printed repr(r.outputs[0].text) for each configuration.
REF_OUTPUT_COMPILE = {
    (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        1,
        False,
        1,
        0,
    ): " The image is a promotional graphic for the Qualcomm® Cloud AI 100, a purpose-built device designed for high-performance, low-power AI processing in both the cloud and edge environments. The graphic features two main components of the Qualcomm® Cloud AI 100:\n\n1. **PCIe Card**: On the left side of the image, there is a PCIe card. This card is designed to be inserted into a PCIe slot on a computer system, allowing it to connect to the system's memory and processing power. The PCIe card is depicted with a sleek, modern design, emphasizing its role in high-performance computing.\n\n2. **",
}

# Performance references.
# PERF_TOLERANCE is the maximum allowed delta from the stored reference value.
# For example, 0.08 means the measured value must stay within 8% of reference.
PERF_TOLERANCE = 0.08
REF_PERF = {
    # Key format:
    #   (model_name, tp_size, gen_len, enable_prefix_caching, qccl_enabled,
    #    sdpa_decode, num_images, prompt_type)
    #
    # num_images:        number of images in the prompt
    # prompt_type:       "describe" (short inline prompt) or "video_prompt" (long structured prompt)
    ("OpenGVLab/InternVL2_5-38B", 4, 128, True, 1, 1, 1, "describe"): {
        "prompt_processing": 16.77,
    },
    ("Qwen/Qwen2.5-VL-3B-Instruct", 1, 128, True, 1, 1, 3, "describe"): {
        "prompt_processing": 9.19,  # TODO: Increasing tolerance due to variation seen in CI.
    },
    ("Qwen/Qwen2.5-VL-32B-Instruct", 4, 128, True, 1, 1, 3, "describe"): {
        "prompt_processing": 12.31,  # TODO: Increasing tolerance due to variation seen in CI.
    },
    ("Qwen/Qwen2.5-VL-32B-Instruct", 4, 128, True, 1, 1, 1, "video_prompt"): {
        "prompt_processing": 7.71,
    },
}

perf_summary_vlm = []
PERF_SUMMARY_FILE = Path(
    os.environ.get("VLM_PERF_SUMMARY_FILE", "ciLogs_vlm_perf_summary.txt")
)


def format_vlm_perf_summary_table():
    if not perf_summary_vlm:
        return None, None

    table = [
        [
            entry["Model"],
            f'{entry["TP"]}',
            f'{entry["Total TTFT"]:.2f}',
            f'{entry["Rendering Prompts"]:.2f}',
            f'{entry["Processed Prompts"]:.2f}',
            f'{entry["Ref Processed Prompts"]:.2f}',
            f'{entry["Processed Prompts Diff %"]:+.2f}%',
        ]
        for entry in perf_summary_vlm
    ]
    headers = [
        "Model",
        "TP",
        "Total TTFT",
        "Rendering Prompts",
        "Processed Prompts",
        "Ref Processed Prompts",
        "Processed Prompts Diff %",
    ]
    return headers, table


def append_vlm_perf_summary_to_file():
    headers, table = format_vlm_perf_summary_table()
    if not table:
        return

    summary_exists = PERF_SUMMARY_FILE.exists() and PERF_SUMMARY_FILE.stat().st_size > 0
    PERF_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PERF_SUMMARY_FILE.open("a", encoding="utf-8") as f:
        if not summary_exists:
            f.write("=== VLM Performance Summary Table ===\n")
            f.write(tabulate(table, headers=headers, tablefmt="grid"))
            f.write("\n")
        else:
            block = tabulate(table, headers=headers, tablefmt="grid").split("\n", 2)[2]
            f.write(block + "\n")


@pytest.fixture(scope="session")
def print_vlm_perf_summary_after_all_tests(request):
    def teardown():
        print_vlm_perf_summary()
        append_vlm_perf_summary_to_file()

    request.addfinalizer(teardown)


def print_vlm_perf_summary():
    headers, table = format_vlm_perf_summary_table()
    if not table:
        print("No VLM performance data collected.")
        return

    print("\n\n=== VLM Performance Summary Table ===")
    print(tabulate(table, headers=headers, tablefmt="grid"))


def assert_perf_within_tolerance(
    metric_name: str, measured_value: float, ref_value: float
):
    # Compare a measured performance metric against its reference value and
    # fail the test if the deviation is greater than PERF_TOLERANCE.
    # If the measured value is better (lower) than the reference, skip the
    # assertion since a negative delta indicates a performance improvement.
    if ref_value <= 0:
        pytest.fail(f"Reference value for {metric_name} must be > 0, got {ref_value}")

    delta = measured_value - ref_value
    if delta < 0:
        # Performance improved — no need to assert.
        return

    delta_pct = delta / ref_value
    assert delta_pct <= PERF_TOLERANCE, (
        f"Performance regression for {metric_name}: measured={measured_value}, "
        f"reference={ref_value}, delta_pct={delta_pct:.2%}, "
        f"allowed={PERF_TOLERANCE:.2%}"
    )


def extract_processed_prompts_time(output: str) -> tuple[float, float]:
    """
    Extract timing values from vLLM tqdm output.

    tqdm reports the per-iteration rate as "s/it" when the average iteration
    takes >= 1s, but switches to "it/s" for sub-second iterations (no
    fractional "s/it" is ever printed). Since these bars always process a
    single request (n=1 total), the "it/s" case is inverted (elapsed = 1 /
    rate) to recover the exact sub-second duration instead of treating it as
    unparsed/zero.

    Examples:
        Rendering prompts: 100%|...| 1/1 [00:10<00:00, 10.86s/it]
        Processed prompts: 100%|...| 1/1 [01:08<00:00, 68.19s/it, est. speed input: 1.64 toks/s, output: 1.88 toks/s]
        Processed prompts: 100%|...| 1/1 [00:00<00:00,  1.06it/s, est. speed input: 90.98 toks/s, output: 1.06 toks/s]

    Returns:
        (rendering_prompts_time, processed_prompts_time)
    """
    print(f"VLLM Profiling: {output}")

    def _extract_seconds(label: str) -> float:
        pattern = re.compile(
            rf"{label}:.*?\[(?:[^,\]]*),\s*(?P<rate>\d+(?:\.\d+)?)\s*(?P<unit>s/it|it/s)\b"
        )
        match = pattern.search(output)
        if not match:
            print(f"Could not find `{label}` timing in captured vLLM output.")
            return 0.0

        rate = float(match.group("rate"))
        if match.group("unit") == "it/s":
            return 1.0 / rate if rate > 0 else 0.0
        return rate

    rendering_prompts_time = _extract_seconds("Rendering prompts")
    processed_prompts_time = _extract_seconds("Processed prompts")

    return rendering_prompts_time, processed_prompts_time


def add_vlm_perf_summary(
    model_name: str,
    tp_size: int,
    total_ttft_time: float,
    rendering_prompts_time: float,
    processed_prompts_time: float,
    ref_processed_prompts_time: float,
):
    processed_prompts_diff_pct = (
        (processed_prompts_time - ref_processed_prompts_time)
        / ref_processed_prompts_time
        * 100
    )

    perf_summary_vlm.append(
        {
            "Model": model_name,
            "TP": tp_size,
            "Total TTFT": total_ttft_time,
            "Rendering Prompts": rendering_prompts_time,
            "Processed Prompts": processed_prompts_time,
            "Ref Processed Prompts": ref_processed_prompts_time,
            "Processed Prompts Diff %": processed_prompts_diff_pct,
        }
    )


@pytest.mark.parametrize("model_name", VLM_MODELS)
@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("gen_len", [128])
@pytest.mark.parametrize("enable_prefix_caching", [True, False])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("env_sdpa_decode", ["0", "1"], indirect=True)
def test_vlm_vllm(
    env_sdpa_decode,
    model_name: str,
    tp_size: int,
    gen_len: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
):
    """Validate VLM output accuracy and basic performance against references."""
    # Runtime knobs are read from the environment so that the same test logic
    # can validate multiple backend/runtime configurations.
    qccl_enabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))
    sdpa_decode = int(os.getenv("QAIC_SDPA_DECODE", 0))

    # Garbage collect so that RAM is freed for RAM limited host.
    gc.collect()

    # Configure compilation for QAIC when not in eager mode.
    # Uses STOCK_TORCH_COMPILE mode with the PyTorch inductor backend and
    # custom_ops=["all"] to enable all vLLM custom ops during compilation.
    #
    # Backend behavior differences in vLLM:
    #   - "inductor" backend: vLLM defaults to custom_ops='none' (disables
    #     all vLLM custom ops) since inductor cannot lower them. We override
    #     with custom_ops=["all"] to match the qaic-inference behavior.
    #   - "qaic-inference" backend: vLLM defaults to custom_ops='all' and
    #     enables fusions (fuse_norm_quant, fuse_act_quant) automatically.
    #
    # NOTE: The vLLM warning "Inductor compilation was disabled by user
    # settings" is expected and harmless with STOCK_TORCH_COMPILE mode — it
    # only means vLLM's piecewise/VLLM_COMPILE-specific optimizations won't
    # apply. The torch.compile pipeline still executes normally.
    #
    # NOTE: QAIC_SDPA_DECODE=1 must be set when using the inductor backend
    # to avoid a FakeTensor error for ops.C_.cpu_attention_with_kv_cache
    # during graph tracing (the operator lacks a meta kernel registration).
    cc = None
    if not enforce_eager:
        cc = CompilationConfig(
            mode=CompilationMode.STOCK_TORCH_COMPILE,  # standard PT2 compile
            backend="inductor",  # PyTorch inductor backend
            custom_ops=[
                "all"
            ],  # enable all vLLM custom ops (matches qaic-inference behavior)
        )

    max_model_len = 16 * 1024
    kv_cache_size = 1024 * 1024 * 1024 * 2  # 2GB KV cache memory
    trust_remote_code = False
    mm_processor_args = None

    # Prompt and processor settings are model-family specific because each
    # VLM expects a different image placeholder/token format.
    if "Qwen" in model_name:
        prompt = "USER: <|vision_start|><|image_pad|><|vision_end|>\nDescribe the image in detail.\nASSISTANT:"
        mm_processor_args = {
            "max_pixels": 100
            * 28
            * 28  # max visual tokens per image each visual token is 28*28.
        }
    elif "InternVL" in model_name:
        prompt = "USER: <image>\nDescribe the image in detail.\nASSISTANT:"
        trust_remote_code = True
    else:
        pytest.skip(f"Describe prompt for model {model_name} not defined!  Exiting...")

    # Load the local test image used for all accuracy/performance checks.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "Cloud_AI_100.jpeg")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}. Exiting...")

    try:
        img = Image.open(img_path)
    except Exception as e:
        pytest.skip(f"Failed to open test image: {e}")

    # Create the vLLM engine with deterministic decoding settings.
    # skip_mm_profiling=True avoids running dummy multimodal encoder forward
    # passes during memory profiling, which significantly reduces startup time
    # for VLM models. This is safe when kv_cache_memory_bytes is set
    # explicitly (bypasses automatic memory calculation).
    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        kv_cache_memory_bytes=kv_cache_size,
        trust_remote_code=trust_remote_code,
        mm_processor_kwargs=mm_processor_args,
        compilation_config=cc,
        skip_mm_profiling=True,
        limit_mm_per_prompt={
            "image": {"count": 16, "width": 512, "height": 512},
            "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
        },
    )

    # First pass: measure approximate TTFT using a single generated token.
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start_time = time.perf_counter()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img],
                },
            },
            sampling_params=SamplingParams(
                seed=RNG_SEED,
                temperature=0.0,
                max_tokens=1,
            ),
        )
    end_time = time.perf_counter()
    ttft_time = end_time - start_time
    ttft_rendering_prompts_time, ttft_processed_prompts_time = (
        extract_processed_prompts_time(
            stdout_buffer.getvalue() + stderr_buffer.getvalue()
        )
    )

    # Second pass: run the full generation used for output validation and for
    # aggregate timing/decode throughput measurement.
    sampling_param = SamplingParams(
        seed=RNG_SEED,
        temperature=0.0,
        max_tokens=gen_len,
    )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start_time = time.perf_counter()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        results = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img],
                },
            },
            sampling_params=sampling_param,
        )
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_rendering_prompts_time, total_processed_prompts_time = (
        extract_processed_prompts_time(
            stdout_buffer.getvalue() + stderr_buffer.getvalue()
        )
    )

    # Validate every returned result. In practice this test uses a single
    # prompt, but iterating keeps the code aligned with the generate() API.
    for r in results:
        print(
            f"Model:{model_name}, TP_SIZE:{tp_size}, KV_CACHE_SIZE (MB):{kv_cache_size/(1024*1024)}, QCCL:{qccl_enabled}, SDPA_DECODE:{sdpa_decode}, PREFIX_CACHING: {enable_prefix_caching}\n"
        )
        print(repr(r.outputs[0].text))
        decode_tokens = len(r.outputs[0].token_ids)
        total_prompt_tokens = len(r.prompt_token_ids)
        text_only_tokens = len(llm.get_tokenizer().encode(prompt))
        visual_tokens_in_prompt = total_prompt_tokens - text_only_tokens
        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Text Tokens: {text_only_tokens}")
        print(f"Visual Tokens in prompt: {visual_tokens_in_prompt}")
        print(f"Output Token shape {decode_tokens}")

        print(
            f"TTFT: Rendering Prompts: {ttft_rendering_prompts_time:.2f}s, Processed Prompt: {ttft_processed_prompts_time:.2f}s, Counter: {ttft_time:.2f}s"
        )
        print(
            f"Total: Rendering Prompts: {total_rendering_prompts_time:.2f}s, Processed Prompt: {total_processed_prompts_time:.2f}s, Counter: {total_time:.2f}s"
        )
        print(
            f"Note: If enable_prefix_caching is True, then decode time is total_time else it is total_time - ttft_time"
        )

        decode_time = (
            total_processed_prompts_time
            if enable_prefix_caching
            else (total_processed_prompts_time - ttft_processed_prompts_time)
        )
        decode_tokens_per_second = (
            (decode_tokens - 1) / decode_time if decode_time > 0 else 0
        )
        print(f"Decode Tokens/Second: {decode_tokens_per_second:.2f}")

        # Accuracy references are keyed by the configuration dimensions that
        # are expected to affect deterministic text output.
        accuracy_key = (
            model_name,
            tp_size,
            enable_prefix_caching,
            qccl_enabled,
            sdpa_decode,
        )
        ref_output = REF_OUTPUT.get(accuracy_key)
        if ref_output is None:
            print(
                f"Missing accuracy reference for config(model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode): {accuracy_key}"
            )
        else:
            assert (
                ref_output == r.outputs[0].text
            ), f"Output mismatch for config(model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode): {accuracy_key}"
            print(
                f"Output matching with reference for config(model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode): {accuracy_key}"
            )


@pytest.mark.parametrize("model_name", VLM_MODELS)
@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("num_images", [1, 3])
@pytest.mark.parametrize("prompt_type", ["describe", "video_prompt"])
@pytest.mark.parametrize("gen_len", [128])
@pytest.mark.parametrize("enable_prefix_caching", [True, False])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("env_sdpa_decode", ["0", "1"], indirect=True)
def test_vlm_vllm_perf(
    env_sdpa_decode,
    print_vlm_perf_summary_after_all_tests,
    model_name: str,
    tp_size: int,
    num_images: int,
    prompt_type: str,
    gen_len: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
):
    """Validate VLM TTFT performance against references."""
    # Runtime knobs are read from the environment so that the same test logic
    # can validate multiple backend/runtime configurations.
    qccl_enabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))
    sdpa_decode = int(os.getenv("QAIC_SDPA_DECODE", 0))

    # Garbage collect so that RAM is freed for RAM limited host.
    gc.collect()

    max_model_len = 16 * 1024
    kv_cache_size = 1024 * 1024 * 1024 * 2  # 2GB KV cache memory
    trust_remote_code = False
    mm_processor_args = None
    model_prompt = {}

    # Load the local test image used for all accuracy/performance checks.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "Cloud_AI_100.jpeg")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}. Exiting...")

    try:
        img = Image.open(img_path)
    except Exception as e:
        pytest.skip(f"Failed to open test image: {e}")

    # Prompt and processor settings are model-family specific because each
    # VLM expects a different image placeholder/token format.
    if "Qwen" in model_name:
        image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        prompt = (
            "USER: "
            + (image_placeholder * num_images)
            + f"\n{TEXT_PROMPT[prompt_type]}\nASSISTANT:"
        )
        model_prompt = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [img] * num_images,
            },
        }

        mm_processor_args = {
            "max_pixels": (1000 if prompt_type == "describe" else 400)
            * 28
            * 28  # max visual tokens per image each visual token is 28*28.
        }
    elif "InternVL" in model_name:
        # Keeping number of images as 1 for InternVL Model as variations is tested for Qwen only.
        prompt = "USER: <image>\n" + TEXT_PROMPT[prompt_type] + "\nASSISTANT:"
        trust_remote_code = True
        model_prompt = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": [img],
            },
        }

    else:
        pytest.skip(f"Describe prompt for model {model_name} not defined!  Exiting...")

    # Create the vLLM engine with deterministic decoding settings.
    # skip_mm_profiling=True avoids running dummy multimodal encoder forward
    # passes during memory profiling, which significantly reduces startup time
    # for VLM models. This is safe when kv_cache_memory_bytes is set
    # explicitly (bypasses automatic memory calculation).
    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        kv_cache_memory_bytes=kv_cache_size,
        trust_remote_code=trust_remote_code,
        mm_processor_kwargs=mm_processor_args,
        skip_mm_profiling=True,
        limit_mm_per_prompt={
            "image": {"count": 16, "width": 512, "height": 512},
            "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
        },
    )

    # First pass: measure TTFT using a single generated token.
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    ttft_wall_start = time.perf_counter()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        ttft_result = llm.generate(
            model_prompt,
            sampling_params=SamplingParams(
                seed=RNG_SEED,
                temperature=0.0,
                max_tokens=1,
            ),
        )
    ttft_wall_end = time.perf_counter()
    ttft_wall_time = ttft_wall_end - ttft_wall_start
    ttft_rendering_prompts_time, ttft_processed_prompts_time = (
        extract_processed_prompts_time(
            stdout_buffer.getvalue() + stderr_buffer.getvalue()
        )
    )

    # Second pass: measure decode throughput over a full generation (gen_len tokens).
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    decode_wall_start = time.perf_counter()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        decode_result = llm.generate(
            model_prompt,
            sampling_params=SamplingParams(
                seed=RNG_SEED,
                temperature=0.0,
                max_tokens=gen_len,
            ),
        )
    decode_wall_end = time.perf_counter()
    decode_wall_time = decode_wall_end - decode_wall_start
    decode_rendering_prompts_time, decode_processed_prompts_time = (
        extract_processed_prompts_time(
            stdout_buffer.getvalue() + stderr_buffer.getvalue()
        )
    )

    for r in ttft_result:
        print(
            f"Model:{model_name}, TP_SIZE:{tp_size}, "
            f"num_images:{num_images}, prompt_type:{prompt_type}, "
            f"KV_CACHE_SIZE (MB):{kv_cache_size/(1024*1024)}, "
            f"QCCL:{qccl_enabled}, SDPA_DECODE:{sdpa_decode}, PREFIX_CACHING:{enable_prefix_caching}\n"
        )
        total_prompt_tokens = len(r.prompt_token_ids)
        text_only_tokens = len(llm.get_tokenizer().encode(model_prompt["prompt"]))
        visual_tokens_in_prompt = total_prompt_tokens - text_only_tokens
        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Text Tokens: {text_only_tokens}")
        print(f"Visual Tokens in prompt: {visual_tokens_in_prompt}")
        print(f"TTFT Wall Time: {ttft_wall_time:.2f}s")
        print(f"TTFT Rendering Prompts Time: {ttft_rendering_prompts_time:.2f}s")
        print(f"TTFT Processed Prompts Time: {ttft_processed_prompts_time:.2f}s")

    for r in decode_result:
        decode_tokens = len(r.outputs[0].token_ids)
        decode_time = (
            decode_processed_prompts_time
            if enable_prefix_caching
            else (decode_processed_prompts_time - ttft_processed_prompts_time)
        )
        decode_tokens_per_second = (
            (decode_tokens - 1) / decode_time if decode_time > 0 else 0
        )
        print(repr(r.outputs[0].text))
        print(f"Output Token shape {decode_tokens}")
        print(
            f"Decode pass: Wall Time: {decode_wall_time:.2f}s, "
            f"Rendering Prompts Time: {decode_rendering_prompts_time:.2f}s, "
            f"Processed Prompts Time: {decode_processed_prompts_time:.2f}s"
        )
        print(f"Decode Tokens/Second: {decode_tokens_per_second:.2f}")

    # Performance references are keyed by the full runtime configuration.
    perf_key = (
        model_name,
        tp_size,
        gen_len,
        enable_prefix_caching,
        qccl_enabled,
        sdpa_decode,
        num_images,
        prompt_type,
    )
    ref_perf = REF_PERF.get(perf_key)
    if ref_perf is None:
        print(
            f"Missing performance reference for config"
            f"(model_name, tp_size, gen_len, enable_prefix_caching, qccl_enabled, "
            f"sdpa_decode, num_images, prompt_type): {perf_key}"
        )
    else:
        assert_perf_within_tolerance(
            "prompt_processing",
            ttft_processed_prompts_time,
            ref_perf["prompt_processing"],
        )
        add_vlm_perf_summary(
            model_name,
            tp_size,
            ttft_wall_time,
            ttft_rendering_prompts_time,
            ttft_processed_prompts_time,
            ref_perf["prompt_processing"],
        )
        print(
            f"Performance matching with reference for config"
            f"(model_name, tp_size, gen_len, enable_prefix_caching, qccl_enabled, "
            f"sdpa_decode, num_images, prompt_type): {perf_key}"
        )


def test_vlm_vllm_compile():
    """Exercise the torch.compile path for vLLM on Qwen2.5-VL-3B (enforce_eager=False)."""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    tp_size = 1
    gen_len = 128
    # Prefix caching is disabled for dynamo caching — without this the cache
    # will grow unboundedly and caching will fail.
    enable_prefix_caching = False
    enforce_eager = False

    os.environ["QAIC_SDPA_DECODE"] = "0"
    qccl_enabled = int(os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0))
    sdpa_decode = int(os.getenv("QAIC_SDPA_DECODE", 0))

    gc.collect()

    cc = CompilationConfig(
        mode=CompilationMode.STOCK_TORCH_COMPILE,
        backend="inductor",
        custom_ops=["all"],
    )

    max_model_len = 16 * 1024
    kv_cache_size = 1024 * 1024 * 1024 * 2  # 2GB KV cache memory

    prompt = "USER: <|vision_start|><|image_pad|><|vision_end|>\nDescribe the image in detail.\nASSISTANT:"
    mm_processor_args = {
        "max_pixels": 100 * 28 * 28,
    }

    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "Cloud_AI_100.jpeg")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found at {img_path}. Exiting...")

    try:
        img = Image.open(img_path)
    except Exception as e:
        pytest.skip(f"Failed to open test image: {e}")

    llm = LLM(
        model=model_name,
        dtype="float16",
        seed=RNG_SEED,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        enable_prefix_caching=enable_prefix_caching,
        kv_cache_memory_bytes=kv_cache_size,
        trust_remote_code=False,
        mm_processor_kwargs=mm_processor_args,
        compilation_config=cc,
        skip_mm_profiling=True,
        limit_mm_per_prompt={
            "image": {"count": 16, "width": 512, "height": 512},
            "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
        },
    )

    # First pass: measure approximate TTFT using a single generated token.
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start_time = time.perf_counter()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img],
                },
            },
            sampling_params=SamplingParams(
                seed=RNG_SEED,
                temperature=0.0,
                max_tokens=1,
            ),
        )
    end_time = time.perf_counter()
    ttft_time = end_time - start_time
    ttft_rendering_prompts_time, ttft_processed_prompts_time = (
        extract_processed_prompts_time(
            stdout_buffer.getvalue() + stderr_buffer.getvalue()
        )
    )

    # Second pass: full generation for output validation.
    sampling_param = SamplingParams(
        seed=RNG_SEED,
        temperature=0.0,
        max_tokens=gen_len,
    )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start_time = time.perf_counter()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        results = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [img],
                },
            },
            sampling_params=sampling_param,
        )
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_rendering_prompts_time, total_processed_prompts_time = (
        extract_processed_prompts_time(
            stdout_buffer.getvalue() + stderr_buffer.getvalue()
        )
    )

    for r in results:
        print(
            f"Model:{model_name}, TP_SIZE:{tp_size}, KV_CACHE_SIZE (MB):{kv_cache_size/(1024*1024)}, "
            f"QCCL:{qccl_enabled}, SDPA_DECODE:{sdpa_decode}, PREFIX_CACHING:{enable_prefix_caching}, "
            f"ENFORCE_EAGER:{enforce_eager}\n"
        )
        print(repr(r.outputs[0].text))
        decode_tokens = len(r.outputs[0].token_ids)
        total_prompt_tokens = len(r.prompt_token_ids)
        text_only_tokens = len(llm.get_tokenizer().encode(prompt))
        visual_tokens_in_prompt = total_prompt_tokens - text_only_tokens
        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Text Tokens: {text_only_tokens}")
        print(f"Visual Tokens in prompt: {visual_tokens_in_prompt}")
        print(f"Output Token shape {decode_tokens}")

        print(
            f"TTFT: Rendering Prompts: {ttft_rendering_prompts_time:.2f}s, Processed Prompt: {ttft_processed_prompts_time:.2f}s, Counter: {ttft_time:.2f}s"
        )
        print(
            f"Total: Rendering Prompts: {total_rendering_prompts_time:.2f}s, Processed Prompt: {total_processed_prompts_time:.2f}s, Counter: {total_time:.2f}s"
        )

        decode_time = total_processed_prompts_time - ttft_processed_prompts_time
        decode_tokens_per_second = (
            (decode_tokens - 1) / decode_time if decode_time > 0 else 0
        )
        print(f"Decode Tokens/Second: {decode_tokens_per_second:.2f}")

        # Accuracy check: validate generated text against stored reference.
        # REF_OUTPUT_COMPILE is keyed separately from REF_OUTPUT because
        # enforce_eager=False (torch.compile) may produce different numerical
        # results than enforce_eager=True (eager), even with temperature=0.0.
        # Key format: (model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode)
        accuracy_key = (
            model_name,
            tp_size,
            enable_prefix_caching,
            qccl_enabled,
            sdpa_decode,
        )
        ref_output = REF_OUTPUT_COMPILE.get(accuracy_key)
        if ref_output is None:
            print(
                f"Missing accuracy reference for compile config"
                f"(model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode): {accuracy_key}"
            )
        else:
            assert ref_output == r.outputs[0].text, (
                f"Output mismatch for compile config"
                f"(model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode): {accuracy_key}"
            )
            print(
                f"Output matching with reference for compile config"
                f"(model_name, tp_size, enable_prefix_caching, qccl_enabled, sdpa_decode): {accuracy_key}"
            )
