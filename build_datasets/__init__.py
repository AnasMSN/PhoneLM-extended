from build_datasets.text_datasets import build_wanjuan_cc, build_fake_texts, build_sky_pile, build_dataset, starcoderdata_dataset
from build_datasets.datasets import TextWrapper
from build_datasets.datasets import CombinedDataset

from build_datasets.text_factory import TextFactory
from functools import partial

from build_datasets.text_datasets import wanjuan_text, get_content
from build_datasets.data_file import DataFileBuilder

TextFactory.register('wanjuan-cc',(
    partial(build_dataset, pattern='*.jsonl', data_type='json'),
    wanjuan_text,
    True,
    ['id', 'content', 'title', 'language', 'date', 'token_num', 'cbytes_num',
     'line_num', 'char_num', 'toxic_score',
     'fluency_score', 'not_ad_score', 'porn_score'],
))

TextFactory.register('SkyPile',(
    partial(build_dataset, pattern='*.jsonl', data_type='json'),
    None,
    False,
    None))

TextFactory.register('StarCoderData',(
    starcoderdata_dataset,
    None,
    False,
    None,
))

TextFactory.register('RefinedWeb',(
    partial(build_dataset, pattern='*.parquet', data_type='parquet'),
    get_content,
    False,
    ['url', 'content', 'timestamp', 'dump', 'segment', 'image_urls']))

TextFactory.register('FakeText', (
    build_fake_texts, None, False, None
))

from build_datasets.data_file import DataFile, PackedDataset
