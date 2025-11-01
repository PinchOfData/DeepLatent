#!/usr/bin/env python
"""
PyPI publishing script for DeepLatent package.
This script builds and uploads the package to PyPI.
"""

import subprocess
import sys
import os
import getpass

def run_command(cmd, description, capture_output=True):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            capture_output=capture_output, 
            text=True
        )
        print(f"✅ {description} completed successfully")
        if capture_output and result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        return False
    
    # Check if build tools are available
    try:
        subprocess.run(["python", "-m", "build", "--help"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: 'build' package not found. Please install it with: pip install build")
        return False
    
    try:
        subprocess.run(["twine", "--help"], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: 'twine' package not found. Please install it with: pip install twine")
        return False
    
    print("✅ Prerequisites check passed")
    return True

def main():
    """Main publishing function."""
    print("🚀 Publishing DeepLatent to PyPI...")
    
    if not check_prerequisites():
        sys.exit(1)
    
    # Clean previous builds
    if os.path.exists("dist"):
        if not run_command("rmdir /s /q dist", "Cleaning previous builds"):
            print("⚠️  Could not clean previous builds, continuing...")
    
    if os.path.exists("build"):
        if not run_command("rmdir /s /q build", "Cleaning build directory"):
            print("⚠️  Could not clean build directory, continuing...")
    
    # Build the package
    if not run_command("python -m build", "Building package"):
        sys.exit(1)
    
    # Check the built package
    if not run_command("twine check dist/*", "Checking built package"):
        sys.exit(1)
    
    # Ask user which repository to upload to
    print("\n📦 Package built successfully!")
    print("Choose upload destination:")
    print("1. TestPyPI (recommended for first time)")
    print("2. PyPI (production)")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter 1 or 2")
    
    if choice == "1":
        repository = "--repository testpypi"
        repo_name = "TestPyPI"
        install_cmd = "pip install --index-url https://test.pypi.org/simple/ deeplatent"
    else:
        repository = ""
        repo_name = "PyPI"
        install_cmd = "pip install deeplatent"
    
    # Upload to PyPI
    upload_cmd = f"twine upload {repository} dist/*"
    print(f"\n🔑 You will be prompted for your {repo_name} credentials...")
    
    if not run_command(upload_cmd, f"Uploading to {repo_name}", capture_output=False):
        sys.exit(1)
    
    print(f"\n🎉 Package successfully published to {repo_name}!")
    print(f"\nUsers can now install it with:")
    print(f"  {install_cmd}")

if __name__ == "__main__":
    main()