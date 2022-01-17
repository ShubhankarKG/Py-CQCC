# Py-CQCC
[![Documentation Status](https://readthedocs.org/projects/py-cqcc/badge/?version=latest)](https://py-cqcc.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/ShubhankarKG/Py-CQCC)](https://github.com/ShubhankarKG/Py-CQCC/blob/master/LICENSE)


Python implementation of CQCC.

This repository is a Python port to the MATLAB package of CQCC available [online](http://audio.eurecom.fr/content/software).

## Documentation

For usage and function documentation, see [documentation](https://py-cqcc.readthedocs.io/en/latest/).

## Usage

```bash
$ pip install Py-CQCC
```

In your `file.py`:

```python
from CQCC import cqcc as cqcc_func

x, sr = librosa.load('file.wav')
B = 96
fmax = sr/2
fmin = fmax//2**9
d = 16
cf = 19
ZsdD = 'ZsdD'

cqcc = cqcc_func.cqcc(x, sr, B, fmax, fmin, d, cf, ZsdD)

print(cqcc)
```

## Future Scope

- [] Simplify API for functions.
- [] Write some tests.


## References

> M. Todisco, H. Delgado, and N. Evans. A New Feature for Automatic Speaker Verification Anti-Spoofing: Constant Q Cepstral Coefficients. Proceedings of ODYSSEY - The Speaker and Language Recognition Workshop, 2016.