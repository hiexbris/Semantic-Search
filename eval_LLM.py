import json
import subprocess

OLLAMA_PATH = r"C:\Users\A-Team\AppData\Local\Programs\ollama\ollama.exe"
MODEL_NAME = "mistral:instruct"
OUTPUT_DIR = "LLM_output"

def ask_llm_only(query):
    prompt = (
        "<|start|>\n"
        "[INST] Answer the following question accurately and directly.[/INST]\n"
        f"{query}\n<|end|>\n"
    )

    try:
        output = subprocess.check_output(
            [OLLAMA_PATH, "run", MODEL_NAME],
            input=prompt.encode("utf-8"),
            stderr=subprocess.DEVNULL,
            timeout=60
        ).decode("utf-8").strip()

        # Strip possible echoed prompt parts
        cleaned = output
        if "[INST]" in cleaned:
            cleaned = cleaned.split("[/INST]")[-1].strip()
        if "<|end|>" in cleaned:
            cleaned = cleaned.split("<|end|>")[0].strip()

        return cleaned

    except subprocess.TimeoutExpired:
        return "Timeout while generating response."
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.decode('utf-8').strip()}"

def load_queries_from_file(file_path="queries.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    return queries


# --- Run Evaluation ---

queries = load_queries_from_file()
for i, query in enumerate(queries):
    answer = ask_llm_only(query)
    with open(f"LLM/{i}.txt", "w", encoding="utf-8") as out_file:
        out_file.write(answer)
        print(f"Done saving in LLM/{i}.txt")
