import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    response = requests.post(
        "http://localhost:8000/query",
        json={"question": args.question, "top_k": args.top_k},
        timeout=180,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

