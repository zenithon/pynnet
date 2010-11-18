import numpy

def load_amat(fname, dtype=numpy.float32):
    rawmat = numpy.loadtxt(fname, dtype=dtype)
    mats = []
    prevcol = 0
    for col in range(rawmat.shape[0]):
        if rawmat[col,0] == -999.:
            mats.append(rawmat[prevcol:col,:])
            prevcol = col+1
    return mats

if __name__ == '__main__':
    import sys
    def shp(m):
        return m.shape
    print map(shp, load_amat(sys.argv[1]))
