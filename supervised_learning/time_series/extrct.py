import zipfile
import sys

def main():
    """extract zip file"""
    file_name = sys.argv[1]
    print("Extracting {}...".format(file_name))
    print(f"we are extracting {file_name}")
    try:
        with zipfile.ZipFile(file_name) as zip_file:
            zip_file.extractall()
        print("Extraction completed.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
