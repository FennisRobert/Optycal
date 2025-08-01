from typing import Iterable, Iterator, TypeVar, T
from itertools import cycle
def loop_iter(items: Iterable[T]) -> Iterator[T]:
    """Loops through an iterable in pairs and ends with the pairing of the first and last

    Args:
        items (Iterable[T]): _description_

    Yields:
        Iterator[T]: _description_
    """    
    for item1, item2 in zip(items[:-1], items[1:]):
        yield item1, item2
    yield items[-1], items[0]

