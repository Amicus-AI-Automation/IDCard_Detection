import os

label_dirs = [
    "train/labels",
    "valid/labels",
    "test/labels"
]

for label_dir in label_dirs:
    for filename in os.listdir(label_dir):
        filepath = os.path.join(label_dir, filename)

        with open(filepath, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = "0"  # change class id to 0
            new_lines.append(" ".join(parts) + "\n")

        with open(filepath, "w") as f:
            f.writelines(new_lines)

print("All classes merged into class 0.")
