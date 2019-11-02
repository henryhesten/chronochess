from collections import defaultdict
from clint.textui import colored

from fantasy_footbot.entities.player import Position


class Team(object):

    VALID_POSITION_COUNTS = {
        Position.goalkeeper: 2,
        Position.defender: 5,
        Position.midfielder: 5,
        Position.forward: 3
    }

    MAX_REAL_LIFE_TEAM_COUNT = 3

    def __init__(self, players):
        assert(len(players) == 15)

        assert(self.correct_position_counts(players))
        assert(self.correct_team_counts(players))
        assert(self.team_cost_within_bounds(players))
        self.players = players
        self.players.sort(key=lambda p: p.ranking_score, reverse=True)

    @classmethod
    def correct_position_counts(cls, players):
        position_counts = defaultdict(int)

        for player in players:
            position_counts[player.position] += 1

        return all([cls.VALID_POSITION_COUNTS[position] == position_counts[position]
                    for position in cls.VALID_POSITION_COUNTS])

    @classmethod
    def correct_team_counts(cls, players):
        team_counts = defaultdict(int)

        for player in players:
            team_counts[player.team] += 1

        return max(team_counts.values()) <= cls.MAX_REAL_LIFE_TEAM_COUNT

    @staticmethod
    def team_cost_within_bounds(players, max_cost=100):
        return sum([p.now_cost/10 for p in players]) <= max_cost

    def __str__(self):
        info_lines = [str(colored.yellow('Total Cost: {:.1f}, Points: {}'.format(
            sum([p.now_cost/10 for p in self.players]),
            sum([self.players[0].total_points * 2] + [p.total_points for p in self.players[1:-4]])
        )))]
        player_lines = [str(colored.red('Players'))]
        for player_no, player in enumerate(self.players):
            player_lines.append("{:>2}. {}".format(player_no+1, player))
        lines = info_lines + player_lines
        return '\n'.join(lines)
