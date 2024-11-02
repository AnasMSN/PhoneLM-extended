from typing import Callable, Dict, TypeVar, Any, Union, List
from datasets import Dataset

T = TypeVar('T', bound=Dataset)


class TextFactory:
    constructor: Dict[
        str, "tuple[Callable[..., T], Callable[..., str], bool, List[str]]"
    ] = {}

    @classmethod
    def register(
        cls,
        name: str,
        constructor: "tuple[Callable[..., T], Union[None,Callable[..., str]], bool, Union[None, List[str]]]",
    ):
        cls.constructor[name] = constructor

    @classmethod
    def build(cls, name: str, *args: Any, **kwargs: Any) -> Dataset:
        build_dataset, get_text, batched, removed_columns = cls.constructor[name]
        kw = {'batched': batched}
        dataset = build_dataset(*args, **kwargs)

        if removed_columns is not None:
            kw['remove_columns'] = removed_columns

        if get_text is not None:
            dataset = dataset.map(get_text, **kw)
        return dataset
