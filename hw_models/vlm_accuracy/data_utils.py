# --------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------

# MIT License

# Copyright (c) 2023 OpenGVLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# LICENCE: https://github.com/OpenGVLab/InternVL/blob/main/LICENSE
# SOURCE:  https://github.com/OpenGVLab/InternVL
# ----------------------------------------------------------------------------------------
"""Utils for data load, save, and process (e.g., prompt construction)"""

import argparse
import json
import re

import yaml

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

CAT_SHORT2LONG = {
    "acc": "Accounting",
    "agri": "Agriculture",
    "arch": "Architecture_and_Engineering",
    "art": "Art",
    "art_theory": "Art_Theory",
    "bas_med": "Basic_Medical_Science",
    "bio": "Biology",
    "chem": "Chemistry",
    "cli_med": "Clinical_Medicine",
    "cs": "Computer_Science",
    "design": "Design",
    "diag_med": "Diagnostics_and_Laboratory_Medicine",
    "econ": "Economics",
    "elec": "Electronics",
    "ep": "Energy_and_Power",
    "fin": "Finance",
    "geo": "Geography",
    "his": "History",
    "liter": "Literature",
    "manage": "Manage",
    "mark": "Marketing",
    "mate": "Materials",
    "math": "Math",
    "mech": "Mechanical_Engineering",
    "music": "Music",
    "phar": "Pharmacy",
    "phys": "Physics",
    "psy": "Psychology",
    "pub_health": "Public_Health",
    "socio": "Sociology",
}


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def load_yaml(file_path):
    with open(file_path, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data["question"]
    o_imgs_paths = []
    for option in data["options"]:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)
    images = [
        data["image_1"],
        data["image_2"],
        data["image_3"],
        data["image_4"],
        data["image_5"],
        data["image_6"],
        data["image_7"],
    ]
    return {
        "id": data["id"],
        "question": question,
        "options": data["options"],
        "answer": data["answer"],
        "image": images,
        "question_type": data["question_type"],
    }


# DATA SAVING
def save_json(filename, ds):
    with open(filename, "w") as f:
        json.dump(ds, f, indent=4)


# Process the input messages to generate prompt for the model.
def get_prompt(messages) -> str:
    """Get the prompt for generation."""
    ## Chat template used for InternVL
    system_prompt = "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    sep = "<|im_end|>\n"

    ret = system_prompt + sep
    for role, message in messages:
        if message:
            if type(message) is tuple:
                message, _, _ = message
            ret += role + message + sep
        else:
            ret += role
    return ret


def str2bool(v):
    if v.lower() in ("yes", "true"):
        return True
    elif v.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")
