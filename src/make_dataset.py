import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Credit%20Default.csv"

def download_dataset():

    os.makedirs("data", exist_ok=True)
    destination = os.path.join("data", "credit_risk_dataset.csv")

    if os.path.exists(destination):

        print("Note: The dataset is already present in the 'data/' folder.")
        return

    print("Downloading the real credit default dataset...")
    urllib.request.urlretrieve(DATA_URL, destination)
    print(f"Download successful! File saved: {destination}")

if __name__ == "__main__":
    download_dataset()