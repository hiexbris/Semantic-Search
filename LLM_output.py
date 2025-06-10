import os
import subprocess
from collections import defaultdict
OLLAMA_PATH = r"C:\Users\A-Team\AppData\Local\Programs\ollama\ollama.exe"  # Update if different

class AnswerSummary:
    def __init__(self, model_name="mistral:instruct"):
        self.model_name = model_name
        os.makedirs("temp", exist_ok=True)
        print(f"\n======= USING MODEL: {self.model_name} with OLLAMA =======\n")

    def format_prompt(self, query, passages):
        combined_passage = "\n".join(passages)
        prompt = (
            f"You are a helpful assistant. Answer the following question using only the text provided.\n\n"
            f"Question: {query}\n\n"
            f"Text:\n{combined_passage}\n\n"
            f"- Only include relevant information.\n"
            f"- No hallucinations or extra information.\n"
            f"- If the answer isn't clearly in the text, say 'Not found'."
        )
        return prompt

 
    def generate_summary(self, prompt):
        try:
            output = subprocess.check_output(
                [OLLAMA_PATH, "run", self.model_name],
                input=prompt.encode("utf-8"),
                stderr=subprocess.DEVNULL,
                timeout=60
            )
            return output.decode("utf-8").strip()
        except subprocess.TimeoutExpired:
            return "Timeout while generating response."
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output.decode('utf-8').strip()}"

    def group_passages_by_title(self, contexts):
        grouped = defaultdict(list)
        for ctx in contexts:
            title = ctx["title"]
            grouped[title].append(ctx["passage"])
        return grouped

    def process_contexts(self, query: str, json_data: dict):
        grouped = self.group_passages_by_title(json_data["merged_contexts"])
        results = []
        output_lines = [f"Query: {query}\n"]

        for link, passages in grouped.items():
            prompt = self.format_prompt(query, passages)
            summary = self.generate_summary(prompt)

            results.append({
                "query": query,
                "link": link,
                "summary": summary
            })

            output_lines.append(f"Link: {link}")
            output_lines.append("Answer:\n" + summary)
            output_lines.append("-" * 80)

        with open("temp/answer.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(output_lines))

        return results
    
    def cleanup(self):
        # Optional: Free up GPU/CPU memory by stopping the Ollama model
        print("Stopping model in Ollama to free memory...")
        try:
            subprocess.run([OLLAMA_PATH, "stop", self.model_name], check=True)
            print(f"Model '{self.model_name}' stopped successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not stop model '{self.model_name}'. Error: {e}")