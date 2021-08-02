"""
forked form https://github.com/iQua/flsim
"""

from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Clients --
        fields = ['num', 'f_max', 'f_min',
                  'cycles_max', 'cycles_min', 'coefficient_max', 'coefficient_min', 'data_size_max', 'data_size_min',
                  'p_max', 'p_min', 'battery_max', 'battery_min']
        defaults = tuple([0] * len(fields))
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        # -- wireless--
        fields = ['bandwidth_max', 'bandwidth_min', 'h', 'variance', ]
        defaults = (5, 1, 1, 1)
        params = [config['wireless'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.wireless = namedtuple('wireless', fields)(*params)

        # -- FL --
        fields = ['epoch','E','T']
        defaults = (100,5,50)
        params = [config['FL'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.FL = namedtuple('FL', fields)(*params)

        # -- DRL--
        fields = ['lambada']
        defaults = (0.5,)
        params = [config['DRL'].get(field, defaults[i])
                  for i, field in enumerate(fields)]

        self.DRL = namedtuple('DRL', fields)(*params)


if __name__ == "__main__":
    config = Config("config.json")
    print(config.DRL)
