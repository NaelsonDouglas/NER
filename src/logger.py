'''
* Defines the standard logging object to be imported accross the project.
Usage:
from logger import log
log.warning('This is a warning')
'''
from rich.logging import RichHandler
import logging
from configs import configs

class _DevWarner:
    """
    Temporary singleton class. Used to allert the dev status only once during the application lifecycle
    """
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance
    def __init__(self):
        self.already_started = True
        if configs.is_dev:
            log.warning('The application is being executed in DEVELOPMENT mode.')


logging.basicConfig(level=configs.LOG_LEVEL, format='%(message)s', datefmt='[%X]', handlers=[RichHandler()])
log = logging.getLogger('rich')
log.setLevel(configs.LOG_LEVEL)

_DevWarner()