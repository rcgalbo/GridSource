#!/usr/bin/env python3
"""
Test runner for GridSource pipeline tests

This script provides a convenient way to run different types of tests
for the GridSource Bank energy banking pipeline.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_unit_tests(verbose=False):
    """Run unit tests"""
    print("🧪 Running unit tests...")
    cmd = ["python", "-m", "pytest", "tests/unit/", "-v" if verbose else ""]
    return subprocess.run([x for x in cmd if x], cwd=project_root)


def run_integration_tests(verbose=False):
    """Run integration tests"""
    print("🔗 Running integration tests...")
    cmd = ["python", "-m", "pytest", "tests/integration/", "-v" if verbose else ""]
    return subprocess.run([x for x in cmd if x], cwd=project_root)


def run_all_tests(verbose=False):
    """Run all tests"""
    print("🚀 Running all tests...")
    cmd = ["python", "-m", "pytest", "tests/", "-v" if verbose else ""]
    return subprocess.run([x for x in cmd if x], cwd=project_root)


def run_tests_with_coverage():
    """Run tests with coverage report"""
    print("📊 Running tests with coverage...")
    cmd = [
        "python", "-m", "pytest", 
        "tests/", 
        "--cov=airflow/dags", 
        "--cov=sagemaker", 
        "--cov-report=html",
        "--cov-report=term"
    ]
    return subprocess.run(cmd, cwd=project_root)


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test function"""
    print(f"🎯 Running specific test: {test_path}")
    cmd = ["python", "-m", "pytest", test_path, "-v" if verbose else ""]
    return subprocess.run([x for x in cmd if x], cwd=project_root)


def check_test_environment():
    """Check if test environment is properly set up"""
    print("🔍 Checking test environment...")
    
    # Check if pytest is installed
    try:
        import pytest
        print("✅ pytest is installed")
    except ImportError:
        print("❌ pytest is not installed. Run: pip install -r tests/requirements.txt")
        return False
    
    # Check if required test dependencies are available
    required_modules = ['pandas', 'numpy', 'requests', 'boto3', 'moto']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} is available")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} is missing")
    
    if missing_modules:
        print(f"\n🔧 Install missing modules: pip install {' '.join(missing_modules)}")
        return False
    
    # Check if test files exist
    test_files = [
        'tests/unit/test_data_extraction.py',
        'tests/unit/test_ml_training.py',
        'tests/integration/test_pipeline_integration.py'
    ]
    
    for test_file in test_files:
        if (project_root / test_file).exists():
            print(f"✅ {test_file} exists")
        else:
            print(f"❌ {test_file} is missing")
            return False
    
    print("\n🎉 Test environment is ready!")
    return True


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='GridSource Pipeline Test Runner')
    parser.add_argument('--type', '-t', 
                       choices=['unit', 'integration', 'all', 'coverage'],
                       default='all',
                       help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check test environment setup')
    parser.add_argument('--test', 
                       help='Run specific test file or function')
    
    args = parser.parse_args()
    
    print("GridSource Bank - Pipeline Test Runner")
    print("=" * 40)
    
    # Check environment if requested
    if args.check:
        if not check_test_environment():
            sys.exit(1)
        return
    
    # Run specific test if requested
    if args.test:
        result = run_specific_test(args.test, args.verbose)
        sys.exit(result.returncode)
    
    # Run tests based on type
    if args.type == 'unit':
        result = run_unit_tests(args.verbose)
    elif args.type == 'integration':
        result = run_integration_tests(args.verbose)
    elif args.type == 'coverage':
        result = run_tests_with_coverage()
    else:  # all
        result = run_all_tests(args.verbose)
    
    # Exit with test result code
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()