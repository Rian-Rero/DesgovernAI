"""Importa os drivers somente quando o veiculo e solicitado."""

__all__ = ['Car']


def __getattr__(name):
    if name == 'Car':
        from .car import Car
        return Car
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
