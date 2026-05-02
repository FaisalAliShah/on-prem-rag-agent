import requests


def main() -> None:
    response = requests.post("http://localhost:8000/ingest", timeout=120)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

