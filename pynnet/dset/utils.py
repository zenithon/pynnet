import os

class MultiFile(object):
    r"""
    File-like object for a sequence of files.

    Only works in read mode.
    """
    def __init__(self, file_list):
        self._file_list = tuple(file_list)
        self._lengths = [None]*len(self.file_list)
        self._cur_index = 0
        self._cur_file = open(self._file_list[0], 'rb')
        self._buf = ''

    def close(self):
        self._file_list = None
        self._lengths = None
        self._buf = None
        self._cur_file.close()

    def _file(self, idx):
        if idx == self._cur_index:
            self._cur_file.seek(0, os.SEEK_SET)
            return
        if self._file_list is None:
            raise ValueError('File is closed')
        if not 0 <= idx < len(self._file_list):
            raise ValueError('Seek out of bounds')
        self._cur_file.close()
        self._cur_index = idx
        self._cur_file = open(self._file_list[self._cur_index], 'rb')

    def _next_file(self):
        if self._cur_index == len(self._file_list) - 1:
            return False
        self._file(self._cur_index + 1)
        return True

    def _read_cur(self, size):
        self._buf += self._cur_file.read(size)


    def _get(self, pos):
        res = self._buf[:pos]
        self._buf = self._buf[pos:]
        return res

    def _buffer(self, size)
        if len(self._buf) < size:
            self._read_cur(size - len(self._buf))
            while len(self._buf) < size and self._next_file():
                self._read_cur(size - len(self._buf))

    def _length(self, idx):
        if self._lengths[idx] is None:
            self._lengths[idx] = os.stat(self._file_list[idx]).st_size
        return self._lengths[idx]

    def read(self, size):
        self._buffer(size)
        return self._get(size)

    def seek(self, pos, whence=os.SEEK_SET):
        self._buf = ''
        if whence == os.SEEK_END:
            idx = len(self._file_list) - 1
            while pos + self._length(idx) < 0:
                pos += self._length(idx)
                idx -= 1
            self._file(idx)
            self._cur_file.seek(pos, os.SEEK_END)
        else:

            if whence == os.SEEK_CUR:
                pos += self.cur_file.tell()
                idx = self._cur_index
            elif whence == os.SEEK_SET:
                idx = 0

            while pos >= self._length(idx):
                pos -= self._length(idx)
                idx += 1
            self._file(idx)
            self._cur_file.seek(pos, os.SEEK_SET)

    def tell(self):
        return sum([self._length(idx) for idx in range(self._cur_index)]) \
            + self._cur_file.tell()
