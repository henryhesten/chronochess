import requests


class FantasyPremierLeagueApi(object):

    FPL_API_ADDRESS = "https://fantasy.premierleague.com/drf/{endpoint}"
    FULL_DATA_ENDPOINT = "bootstrap-static"
    PLAYER_DATA_ENDPOINT = "element-summary/{id}"

    cached_full_api_response = None

    @classmethod
    def _get_full_api_data(cls):
        """
        Gets the main API response the FPL site provides.
        Will be cached for the length of the program command
        """
        if cls.cached_full_api_response is not None:
            return cls.cached_full_api_response
        else:
            response = requests.get(cls.FPL_API_ADDRESS.format(endpoint=cls.FULL_DATA_ENDPOINT))
            cls.cached_full_api_response = response.json()
            return response.json()

    @classmethod
    def get_cutdown_player_data(cls):
        api_data = cls._get_full_api_data()
        return api_data['elements']

    @classmethod
    def get_team_name_mapping(cls):
        api_data = cls._get_full_api_data()
        team_names = {}
        for team in api_data['teams']:
            team_names[team['code']] = team['name']
        return team_names

    @classmethod
    def get_full_player_data(cls, cutdown_player_data: dict):
        player_endpoint = cls.FPL_API_ADDRESS.format(endpoint=cls.PLAYER_DATA_ENDPOINT)
        response = requests.get(player_endpoint.format(id=cutdown_player_data['id']))
        if response.status_code != 200:
            raise Exception("Failed to get full player data")
        player_data = response.json()
        player_data.update(cutdown_player_data)
        return player_data
