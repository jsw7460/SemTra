import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)
