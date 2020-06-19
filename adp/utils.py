# Alias for `funcs` module but deprecated, please use `funcs` instead
import warnings
from .funcs import *
warnings.warn('Please use `funcs` module instead of `utils` module. '
              'Renamed to avoid confusion with `utility` module.')
