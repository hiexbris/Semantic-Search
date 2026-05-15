import os
import subprocess
from collections import defaultdict
OLLAMA_PATH = "ollama"  

class AnswerSummary:
    def __init__(self, model_name="mistral:instruct"): #phi3 if crashes
        self.model_name = model_name
        os.makedirs("temp", exist_ok=True)
        print(f"\n======= USING MODEL: {self.model_name} with OLLAMA =======\n")

    def format_prompt(self, query, passages):
        combined_passage = "\n".join(passages)
        prompt = (
            f"You are a helpful assistant. Answer the following question using text provided and you're knowledge.\n\n"
            f"Question: {query}\n\n"
            f"Text:\n{combined_passage}\n\n"
            f"- Only include relevant information.\n"
        )
        return prompt

    def generate_summary(self, prompt):
        import requests
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        try:
            # 5 minute timeout to accommodate computers running on CPU
            response = requests.post(url, json=data, timeout=300)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"Error: Ollama returned status {response.status_code}. Make sure Ollama app is running."
        except requests.exceptions.Timeout:
            return "Timeout while generating response. Your computer might need more time to process."
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Please make sure the Ollama app is open and running in the background!"

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