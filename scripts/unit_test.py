import os
import subprocess

def unit_test(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Test py file: {file_path}")
                
                try:
                    result = subprocess.run(
                        ["python", file_path],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    print(f"file {file_path} test success:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"file {file_path} test failed:\n{e.stderr}")

if __name__ == "__main__":
    project_directory = "../lib"
    unit_test(project_directory)
    project_directory = "../classical"
    unit_test(project_directory)
