import logging
import os

src_logger = logging.getLogger(__name__)


class PackagePathFilter(logging.Filter):
    def filter(self, record):
        record.pathname = os.path.realpath(record.pathname).replace(os.getcwd(), '')[1:]
        return True


def setup_logging():
    # Message formatter
    msg_format = '%(asctime)s [%(levelname)8s] %(message)s (%(name)s - %(pathname)s:%(lineno)d)'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=msg_format, datefmt=date_format)

    # File Handler
    file_handler = logging.FileHandler('./log/errors.log', encoding='utf-8')
    file_handler.setFormatter(fmt=formatter)
    file_handler.addFilter(PackagePathFilter())
    src_logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=formatter)
    stream_handler.addFilter(PackagePathFilter())
    src_logger.addHandler(stream_handler)

    # Since this is the root src_logger of this project
    src_logger.propagate = False

    # Set the level for the src_logger
    src_logger.setLevel(logging.INFO)


setup_logging()
