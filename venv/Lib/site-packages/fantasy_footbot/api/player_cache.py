import json

import os
import time


# TODO player cache should have an all-or-nothing option for checking the cache rather than for each player


class PlayerCache(object):
    CACHE_ROOT_LOCATION = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cached_api_data")

    @classmethod
    def in_cache(cls, player_id):
        """
        If file for player id is present in the cache, recent to the last two hours, return True.
        Otherwise, return False.
        """
        if not os.path.exists(os.path.join(cls.CACHE_ROOT_LOCATION, str(player_id))):
            return False
        expired = os.path.getmtime(
            os.path.join(cls.CACHE_ROOT_LOCATION, str(player_id))) < time.time() - 7200
        return not expired

    @classmethod
    def get_full_player_data(cls, cutdown_player_data):
        with open(os.path.join(cls.CACHE_ROOT_LOCATION, str(cutdown_player_data['id'])), 'r') as player_file:
            return json.load(player_file)

    @classmethod
    def save_full_player_data(cls, player_id, player_data):
        with open(os.path.join(cls.CACHE_ROOT_LOCATION, str(player_id)), 'w') as player_file:
            json.dump(player_data, player_file, indent=2)


if not os.path.exists(PlayerCache.CACHE_ROOT_LOCATION):
    os.makedirs(PlayerCache.CACHE_ROOT_LOCATION)
