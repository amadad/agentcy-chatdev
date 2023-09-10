# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import argparse
import logging
import os
import sys
import streamlit as st

from camel.typing import ModelType

root = os.path.dirname(__file__)
sys.path.append(root)

from chatdev.chat_chain import ChatChain


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--config', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")
parser.add_argument('--org', type=str, default="DefaultOrganization",
                    help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--task', type=str, default="Develop a basic Gomoku game.",
                    help="Prompt of software")
parser.add_argument('--name', type=str, default="Gomoku",
                    help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--model', type=str, default="GPT_3_5_TURBO",
                    help="GPT Model, choose from {'GPT_3_5_TURBO','GPT_4','GPT_4_32K'}")
args = parser.parse_args()

def main():
    st.title("ChatDev Interface")

    # Configuration options
    config_option = st.selectbox("Choose a config:", ["Default", "scty"])
    org_option = st.text_input("Organization Name:", "DefaultOrganization")
    task_option = st.text_input("Task:", "Marketing activity for ideation.")
    name_option = st.text_input("Campaign Name:", "Gomoku")
    model_option = st.selectbox("GPT Model:", ["GPT_3_5_TURBO", "GPT_4", "GPT_4_32K"])

    if st.button("Start ChatDev"):
        # Placeholder for output
        output_placeholder = st.empty()
        output_placeholder.text("Processing...")

        # ----------------------------------------
        #          Init ChatChain
        # ----------------------------------------

        config_path, config_phase_path, config_role_path = get_config(config_option)
        args2type = {'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO, 'GPT_4': ModelType.GPT_4, 'GPT_4_32K': ModelType.GPT_4_32k}
        chat_chain = ChatChain(config_path=config_path,
                               config_phase_path=config_phase_path,
                               config_role_path=config_role_path,
                               task_prompt=task_option,
                               project_name=name_option,
                               org_name=org_option,
                               model_type=args2type[model_option])

        # ----------------------------------------
        #          Init Log
        # ----------------------------------------
        logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                            format='[%(asctime)s %(levelname)s] %(message)s',
                            datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

        # ----------------------------------------
        #          Pre Processing
        # ----------------------------------------
        chat_chain.pre_processing()

        # ----------------------------------------
        #          Personnel Recruitment
        # ----------------------------------------
        chat_chain.make_recruitment()

        # ----------------------------------------
        #          Chat Chain
        # ----------------------------------------
        chat_chain.execute_chain()

        # ----------------------------------------
        #          Post Processing
        # ----------------------------------------
        chat_chain.post_processing()

      # Update the placeholder with the final output
        output_placeholder.text("This is the final output.")

        # Display the log link
        st.markdown(f"[View Log]({chat_chain.log_filepath})")

    else:
        st.write("This is where the output will be displayed.")

if __name__ == "__main__":
    main()