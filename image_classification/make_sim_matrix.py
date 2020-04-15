import logging
from itertools import combinations_with_replacement
from pathlib import Path
import click
import numpy as np
from tqdm import tqdm
import gensim.downloader as api

from simloss.utils.constants import CIFAR100_CLASSES, CIFAR100_BLACKLIST, SIM_MATRIX_PATH

@click.command()
@click.argument("output_path", type=click.Path(writable=True), default=SIM_MATRIX_PATH, required=False)
def main(output_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info('making similarity matrix for CIFAR100 class names')

    model = api.load('word2vec-google-news-300')

    classes = [c for c in CIFAR100_CLASSES if c not in CIFAR100_BLACKLIST]

    similarities = {frozenset([w1, w2]): model.similarity(w1, w2)
                    for w1, w2 in combinations_with_replacement(classes, 2)}

    logger.info(similarities)

    word_count = len(classes)
    sim_matrix = np.zeros((word_count, word_count))

    for i1, w1 in tqdm(enumerate(classes)):
        for i2, w2 in tqdm(enumerate(classes)):
            sim_matrix[i1, i2] = similarities[frozenset([w1, w2])]

    logger.info(sim_matrix)

    np.save(output_path, sim_matrix)

    logger.info(f'saved matrix to {output_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
