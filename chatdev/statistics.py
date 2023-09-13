import os
import numpy as np

def get_info(dir, log_filepath):
    def safe_open(filepath, mode="r", encoding="utf8"):
        if os.path.exists(filepath):
            return open(filepath, mode, encoding=encoding).read().split("\n")
        else:
            print(f"Warning: File {filepath} not found.")
            return []

    print("dir:", dir)

    version_updates = -1
    num_code_files = -1
    num_png_files = -1
    num_doc_files = -1
    code_lines = -1
    env_lines = -1
    manual_lines = -1
    duration = -1
    num_utterance = -1
    num_reflection = -1
    num_prompt_tokens = -1
    num_completion_tokens = -1
    num_total_tokens = -1

    if os.path.exists(dir):
        filenames = os.listdir(dir)

        num_code_files = len([filename for filename in filenames if filename.endswith(".py")])

        num_png_files = len([filename for filename in filenames if filename.endswith(".png")])

        num_doc_files = 0
        for filename in filenames:
            if filename.endswith(".py") or filename.endswith(".png"):
                continue
            if os.path.isfile(os.path.join(dir, filename)):
                num_doc_files += 1

        lines = safe_open(os.path.join(dir, "meta.txt"))
        version_updates = float([lines[i + 1] for i, line in enumerate(lines) if "Code_Version" in line][0]) + 1 if "meta.txt" in filenames else -1

        lines = safe_open(os.path.join(dir, "requirements.txt"))
        env_lines = len([line for line in lines if len(line.strip()) > 0]) if "requirements.txt" in filenames else -1

        lines = safe_open(os.path.join(dir, "manual.md"))
        manual_lines = len([line for line in lines if len(line.strip()) > 0]) if "manual.md" in filenames else -1

        code_lines = 0
        for filename in filenames:
            if filename.endswith(".py"):
                lines = safe_open(os.path.join(dir, filename))
                code_lines += len([line for line in lines if len(line.strip()) > 0])

        lines = safe_open(log_filepath)
        start_lines = [line for line in lines if "**[Start Chat]**" in line]
        chat_lines = [line for line in lines if "<->" in line]
        num_utterance = len(start_lines) + len(chat_lines)

        sublines = [line for line in lines if line.startswith("prompt_tokens:")]
        if sublines:
            nums = [int(line.split(": ")[-1]) for line in sublines]
            num_prompt_tokens = np.sum(nums)

        sublines = [line for line in lines if line.startswith("completion_tokens:")]
        if sublines:
            nums = [int(line.split(": ")[-1]) for line in sublines]
            num_completion_tokens = np.sum(nums)

        sublines = [line for line in lines if line.startswith("total_tokens:")]
        if sublines:
            nums = [int(line.split(": ")[-1]) for line in sublines]
            num_total_tokens = np.sum(nums)

        num_reflection = len([line for line in lines if "on : Reflection" in line])

    cost = 0.0
    if num_png_files != -1:
        cost += num_png_files * 0.016
    if num_prompt_tokens != -1:
        cost += num_prompt_tokens * 0.003 / 1000.0
    if num_completion_tokens != -1:
        cost += num_completion_tokens * 0.004 / 1000.0

    info = "\n\n💰**cost**=${:.6f}\n\n🔨**version_updates**={}\n\n📃**num_code_files**={}\n\n🏞**num_png_files**={}\n\n📚**num_doc_files**={}\n\n📃**code_lines**={}\n\n📋**env_lines**={}\n\n📒**manual_lines**={}\n\n🗣**num_utterances**={}\n\n🤔**num_self_reflections**={}\n\n❓**num_prompt_tokens**={}\n\n❗**num_completion_tokens**={}\n\n🌟**num_total_tokens**={}" \
        .format(cost,
                version_updates,
                num_code_files,
                num_png_files,
                num_doc_files,
                code_lines,
                env_lines,
                manual_lines,
                num_utterance,
                num_reflection,
                num_prompt_tokens,
                num_completion_tokens,
                num_total_tokens)

    return info