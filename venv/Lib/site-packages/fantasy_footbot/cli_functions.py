import argparse
import json
import sys

import os
from clint.textui import progress

from fantasy_footbot import ranking_functions, building_functions
from fantasy_footbot.api import FantasyPremierLeagueApi
from fantasy_footbot.api.player_cache import PlayerCache
from fantasy_footbot.entities import Team, Player


class FPLException(Exception):
    pass


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Fantasy Football Team Helper')
    parser.add_argument('command', type=str,
                        help='Commands for the fpl scout',
                        choices=['team', 'scout'])
    parser.add_argument('-tf', '--teamfile', type=str,
                        help='reference a custom location for a Teamfile',
                        default=os.path.join(os.path.dirname(__file__), 'Teamfile.json'))
    parser.add_argument('-rf', '--rankingfunc', type=str,
                        help='choose the function to rank players with',
                        choices=['ppg', 'rach'],
                        default='ppg')
    parser.add_argument('-bf', '--buildingfunc', type=str,
                        help='choose the function to build a team of players with',
                        choices=['lp_max'],
                        default='lp_max')
    return parser


def cli_main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = _build_arg_parser()
    parsed_args = parser.parse_args(args)

    if not os.path.exists(parsed_args.teamfile):
        _create_new_teamfile(parsed_args.teamfile)

    if parsed_args.command == "team":
        fpl_team(parsed_args)
    elif parsed_args.command == "scout":
        fpl_scout(parsed_args)


def _create_new_teamfile(path):
    team_data_template = {
        "players": {
            "active": [],
            "bench": []
        },
        "bank": 100,
        "points": 0
    }
    with open(path, 'w') as new_teamfile:
        json.dump(team_data_template, new_teamfile, indent=2)


def fpl_team(args):
    with open(args.teamfile, 'r') as teamfile:
        team_data = json.load(teamfile)
    team = Team(team_data)
    print(team)


def fpl_scout(args):
    players = _get_players_from_api()
    ranking_func = _get_ranking_func(args.rankingfunc)
    _rank_players(players, ranking_func)
    players.sort(key=lambda p: p.ranking_score)
    building_func = _get_building_func(args.buildingfunc)
    team = _build_team(players, building_func)
    print(team)


def _get_players_from_api(premier_league_only=True, add_team_names=True):
    all_cutdown_player_data = FantasyPremierLeagueApi.get_cutdown_player_data()
    players = []
    with progress.Bar(label="Loading Player Data: ", expected_size=len(all_cutdown_player_data)) as bar:
        for count, cutdown_player_data in enumerate(all_cutdown_player_data):
            bar.show(count + 1)
            if PlayerCache.in_cache(cutdown_player_data['id']):
                player_data = PlayerCache.get_full_player_data(cutdown_player_data)
            else:
                player_data = FantasyPremierLeagueApi.get_full_player_data(cutdown_player_data)
                PlayerCache.save_full_player_data(player_data['id'], player_data)
            players.append(Player(player_data))

    team_names = FantasyPremierLeagueApi.get_team_name_mapping()

    if premier_league_only:
        players = [p for p in players if p.team_code in team_names.keys()]

    if add_team_names:
        for player in players:
            player.team = team_names[player.team_code]

    return players


def _get_ranking_func(ranking_func_str):
    if ranking_func_str == "ppg":
        ranking_func = ranking_functions.points_per_game_rank
    elif ranking_func_str == "rach":
        ranking_func = ranking_functions.rach_rank
    else:
        raise FPLException("Unrecognised ranking function")
    return ranking_func


def _rank_players(players, ranking_func):
    scores = [ranking_func(player) for player in players]
    for player, score in zip(players, scores):
        player.ranking_score = score


def _get_building_func(building_func_str):
    if building_func_str == "lp_max":
        building_func = building_functions.lp_max_score_build
    else:
        raise FPLException("Unrecognised building function")
    return building_func


def _build_team(players, building_func):
    selected_players = building_func(players)
    return Team(selected_players)


# get all the players
# load the data into Player objects
# compute a score of goodness for each Player
# come up with a team which fits within the Fantasy Football requirements
# save the Team to file

# li_main(['scout', '-rf', 'rach'])
