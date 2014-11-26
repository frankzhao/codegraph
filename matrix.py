a = [[1,2],
     [3,4]]

b = [[2],[3]]

out = [[0],[0]]

def matrix(a,b):
    for arow in range(len(a)):
        for bcol in range(len(b[0])):
            for brow in range(len(b)):
                out[arow][bcol] += a[arow][brow] * b[brow][bcol]
    for e in out:
        print(e)
