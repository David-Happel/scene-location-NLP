def int_to_one_hot(li):
    res = list()
    n = li.max()
    for i in li:
        encoding = [0 for _ in range(n+1)]
        encoding[i] = 1
        res.append(encoding)
    return res

def one_hot_to_int(li):
    res = list()
    for i in li:
        category = i.index(1)
        res.append(category)
    return res
