import os
import requests

def download_files_from_urls(urls_file, output_dir):
    with open(urls_file, 'r') as file:
        for url in file:
            url = url.strip()  # Remove leading/trailing whitespaces
            if url:  # Check if the line is not empty
                download_file(url, output_dir)

def download_file(url, output_dir):
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    print(f"Downloading {filename} from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename} successfully!")
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")

if __name__ == "__main__":
    urls_file = 'urls.txt'  # Path to the file containing URLs
    output_dir = 'downloads'  # Output directory to save downloaded files

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    download_files_from_urls(urls_file, output_dir)
