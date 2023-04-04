import requests


def store(filepath: str) -> None:
    url = "http://127.0.0.1:5000/index"
    with open(filepath, "r") as f:
        passage = f.read()
    headers = {"content-type": "application/json"}
    payload = {"payload": passage, "file": filepath}
    response = requests.post(
        url,
        json=payload,
        headers=headers,
    )
    print(response.text)


def request(query: str) -> str:
    url = "http://127.0.0.1:5000/query"
    headers = {"content-type": "application/json"}
    payload = {"query": query}
    response = requests.post(
        url,
        json=payload,
        headers=headers,
    )
    print(response.text)


if __name__ == "__main__":
    # store("./data/paulgraham_0.txt")
    # store("./data/paulgraham_1.txt")
    # store("./data/paulgraham_2.txt")
    # store("./data/harry_potter_book_1.txt")
    request("Who are harry potter's friends?")
