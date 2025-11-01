#!/usr/bin/env python
"""
Development setup script for DeepLatent package.
Run this script to install the package in development mode.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up DeepLatent for development...")
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Install build tools
    if not run_command("pip install --upgrade pip setuptools wheel build", "Installing build tools"):
        sys.exit(1)
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️  Development dependencies installation failed, continuing...")
    
    print("\n🎉 Development setup completed!")
    print("\nYou can now:")
    print("  - Import deeplatent in Python: 'import deeplatent'")
    print("  - Run tests (if available): 'pytest'")
    print("  - Build package: 'python -m build'")

if __name__ == "__main__":
    main()