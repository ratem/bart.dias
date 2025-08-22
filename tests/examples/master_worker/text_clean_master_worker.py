def normalize_usernames(usernames):
    # Transformação simples por item (string ops funcionam bem no worker)
    cleaned = []
    for u in usernames:
        cleaned.append(u.strip().lower())
    return cleaned


if __name__ == "__main__":
    names = [" Alice ", "BOB", "  Carol  "]
    print(normalize_usernames(names))  # esperado: ['alice', 'bob', 'carol']
