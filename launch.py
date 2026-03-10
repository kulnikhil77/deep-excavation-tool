"""
Launch helper for Deep Excavation Analysis Tool.
Usage: python launch.py
"""

import subprocess
import sys
import os

def check_deps():
    """Check and install missing dependencies."""
    deps = ['streamlit', 'plotly', 'docx', 'numpy', 'matplotlib', 'pandas']
    missing = []
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        # Map import names to pip names
        pip_names = {'docx': 'python-docx'}
        pkgs = [pip_names.get(m, m) for m in missing]
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + pkgs)
        print("Dependencies installed.")
    else:
        print("All dependencies OK.")


def main():
    check_deps()

    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    print(f"\nLaunching Deep Excavation Tool...")
    print(f"Open browser at: http://localhost:8501\n")

    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', app_path,
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
    ])


if __name__ == '__main__':
    main()
