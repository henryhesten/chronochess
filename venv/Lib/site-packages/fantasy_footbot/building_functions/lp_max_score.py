import pulp

from fantasy_footbot.entities.player import Position
from fantasy_footbot.entities.team import Team


def lp_max_score_build(players: list):
    best_team_problem = pulp.LpProblem("best_team", pulp.LpMaximize)

    player_variables = {}
    for player in players:
        player_variables[player.id] = pulp.LpVariable(player.id, lowBound=0, upBound=1, cat='Binary')

    # build objective function
    objective_function = float(players[0].ranking_score) * player_variables[players[0].id]
    for player in players[1:]:
        objective_function = objective_function + float(player.ranking_score) * player_variables[player.id]

    # objective, max ranking_score
    best_team_problem += objective_function

    # constraints

    # cost constraint
    best_team_problem += _build_cost_constraint(players, player_variables)

    # position constraints
    for position in Position:
        best_team_problem += _build_position_constraint(players, position, player_variables)

    # team constraints
    for team in set([p.team for p in players]):
        best_team_problem += _build_team_constraint(players, team, player_variables)

    # TODO captain constraint, consider that captain should be better invested into
    # TODO bench constraint, consider that the bottom 4 players aren't as important

    best_team_problem.solve()

    selected_player_ids = [i.name for i in best_team_problem.variables() if i.varValue == 1]

    # TODO might be a better way of going from ids to chosen players
    return [p for p in players if str(p.id) in selected_player_ids]


def _build_position_constraint(players, position, player_variable_dict):
    # goalkeepers
    players_in_position = [p for p in players if p.position == position]
    constraint = player_variable_dict[players_in_position[0].id]
    for player in players_in_position[1:]:
        constraint = constraint + player_variable_dict[player.id]
    return constraint == Team.VALID_POSITION_COUNTS[position]


def _build_team_constraint(players, team, player_variable_dict):
    players_in_team = [p for p in players if p.team == team]
    constraint = player_variable_dict[players_in_team[0].id]
    for player in players_in_team[1:]:
        constraint = constraint + player_variable_dict[player.id]
    return constraint <= Team.MAX_REAL_LIFE_TEAM_COUNT


def _build_cost_constraint(players, player_variable_dict, team_cost=100.0):
    constraint = float(players[0].now_cost/10) * player_variable_dict[players[0].id]
    for player in players[1:]:
        constraint = constraint + float(player.now_cost/10) * player_variable_dict[player.id]
    return constraint <= team_cost
