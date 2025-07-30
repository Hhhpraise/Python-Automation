import os
import shutil
from pathlib import Path


def organize_downloads():
    # Define file categories inside the function
    FILE_CATEGORIES = {
        "Images": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"],
        "Videos": [".mp4", ".mov", ".avi", ".mkv"],
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".csv"],
        "Audio": [".mp3", ".wav", ".ogg"],
        "Archives": [".zip", ".rar", ".7z"],
        "Code": [".py", ".js", ".html", ".css", ".java"],
        "Executables": [".exe", ".msi", ".dmg"],
    }

    downloads_path = str(Path.home() / "Downloads")

    for folder in FILE_CATEGORIES.keys():
        folder_path = os.path.join(downloads_path, folder)
        os.makedirs(folder_path, exist_ok=True)

    for file in os.listdir(downloads_path):
        file_path = os.path.join(downloads_path, file)

        if os.path.isdir(file_path) or file == __file__:
            continue

        _, extension = os.path.splitext(file)
        extension = extension.lower()

        moved = False
        for category, extensions in FILE_CATEGORIES.items():
            if extension in extensions:
                dest_folder = os.path.join(downloads_path, category)
                dest_path = os.path.join(dest_folder, file)

                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(file_path, dest_path)
                moved = True
                break

        if not moved:
            others_path = os.path.join(downloads_path, "Others")
            os.makedirs(others_path, exist_ok=True)
            shutil.move(file_path, os.path.join(others_path, file))


if __name__ == "__main__":
    organize_downloads()
    print("✅ Downloads folder organized!")