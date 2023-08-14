import subprocess


def run_makefile(target):
    makefile_path = 'Makefile'
    try:
        subprocess.run(["make", "-f", makefile_path, target], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Makefile target '{target}':")
        print(e)
        raise e