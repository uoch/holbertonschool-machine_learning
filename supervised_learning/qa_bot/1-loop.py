#!/usr/bin/python3
""" console """
import sys


def main():
    """loop for questions and answers"""
    while True:
        try:
            sentence = input("Q: ")
            sentence = sentence.lower()
            if sentence in ['exit', 'quit', 'goodbye', 'bye']:
                print("A: Goodbye")
                sys.exit()
            else:
                print("A: ")
        except (KeyboardInterrupt, EOFError):
            print("A: Goodbye")
            sys.exit()
if __name__ == "__main__":
    main()