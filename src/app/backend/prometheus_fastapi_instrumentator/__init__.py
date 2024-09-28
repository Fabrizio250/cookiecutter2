"""
prometheus_fastapi_instrumentator including PR #203
https://github.com/trallnag/prometheus-fastapi-instrumentator/pull/203

Thanks to https://github.com/HadilD/ :)
"""

from .instrumentation import PrometheusFastApiInstrumentator

__version__ = "5.9.1"

Instrumentator = PrometheusFastApiInstrumentator