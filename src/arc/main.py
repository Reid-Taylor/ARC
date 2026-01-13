def main():
    print("Hello from arc!")

if __name__ == "__main__":
    # Updated to use the new testing approach
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        from tests.test_arc_module import run_interactive_test
        run_interactive_test()
    else:
        main()