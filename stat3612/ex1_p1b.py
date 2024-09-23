for i in range(1, 7):
    ans = 15 - 5781/200 * i + 1127/48 * i**2 - 317/40 * i**3 + 293/240 * i**4 - 7/100 * i**5
    print(f"i = {i}, ans = {ans}")
