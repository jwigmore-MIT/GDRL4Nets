# run_all_tests.py
import pytest

def run_tests():
    test_files = [
        "test_MCMH_simulation.py",
        "test_torchrl.py",
        "test_backpressure.py",
        "test_pyg_transformation.py",
    ]

    failed_tests = []

    for test_file in test_files:
        result = pytest.main([test_file])
        if result != 0:
            failed_tests.append(test_file)
            print(f"Tests in {test_file} failed.")
        else:
            print(f"Tests in {test_file} passed.")

    if failed_tests:
        print("\nSummary of failed tests:")
        for test_file in failed_tests:
            print(f"- {test_file}")
        raise Exception("Some tests failed.")
    else:
        print("All tests passed.")

if __name__ == "__main__":
    run_tests()