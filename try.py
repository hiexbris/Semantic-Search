import os

# Create the folder if it doesn't exist
folder_path = "GPT"
os.makedirs(folder_path, exist_ok=True)

# Create empty text files from 0.txt to 48.txt
for i in range(49):
    file_path = os.path.join(folder_path, f"{i}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        pass  # Write nothing (empty file)

print("Created 49 empty files inside GPT/")
