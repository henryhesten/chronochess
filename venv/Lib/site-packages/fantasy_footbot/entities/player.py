import enum


class Position(enum.Enum):
    goalkeeper = 1
    defender = 2
    midfielder = 3
    forward = 4


class Player(object):
    def __init__(self, json_data):
        self.assists = json_data['assists']
        self.bonus = json_data['bonus']
        self.bps = json_data['bps']
        self.chance_of_playing_next_round = json_data['chance_of_playing_next_round']
        self.chance_of_playing_this_round = json_data['chance_of_playing_this_round']
        self.clean_sheets = json_data['clean_sheets']
        self.code = json_data['code']
        self.cost_change_event = json_data['cost_change_event']
        self.cost_change_event_fall = json_data['cost_change_event_fall']
        self.cost_change_start = json_data['cost_change_start']
        self.cost_change_start_fall = json_data['cost_change_start_fall']
        self.creativity = json_data['creativity']
        self.dreamteam_count = json_data['dreamteam_count']
        self.ea_index = json_data['ea_index']
        self.element_type = json_data['element_type']
        self.ep_next = json_data['ep_next']
        self.ep_this = json_data['ep_this']
        self.event_points = json_data['event_points']
        self.explain = json_data['explain']
        self.first_name = json_data['first_name']
        self.fixtures = json_data['fixtures']
        self.fixtures_summary = json_data['fixtures_summary']
        self.form = json_data['form']
        self.goals_conceded = json_data['goals_conceded']
        self.goals_scored = json_data['goals_scored']
        self.history = json_data['history']
        self.history_past = json_data['history_past']
        self.history_summary = json_data['history_summary']
        self.ict_index = json_data['ict_index']
        self.id = json_data['id']
        self.in_dreamteam = json_data['in_dreamteam']
        self.influence = json_data['influence']
        self.loaned_in = json_data['loaned_in']
        self.loaned_out = json_data['loaned_out']
        self.loans_in = json_data['loans_in']
        self.loans_out = json_data['loans_out']
        self.minutes = json_data['minutes']
        self.news = json_data['news']
        self.now_cost = json_data['now_cost']
        self.own_goals = json_data['own_goals']
        self.penalties_missed = json_data['penalties_missed']
        self.penalties_saved = json_data['penalties_saved']
        self.photo = json_data['photo']
        self.points_per_game = json_data['points_per_game']
        self.red_cards = json_data['red_cards']
        self.saves = json_data['saves']
        self.second_name = json_data['second_name']
        self.selected_by_percent = json_data['selected_by_percent']
        self.special = json_data['special']
        self.squad_number = json_data['squad_number']
        self.status = json_data['status']
        self.team = json_data['team']
        self.team_code = json_data['team_code']
        self.threat = json_data['threat']
        self.total_points = json_data['total_points']
        self.transfers_in = json_data['transfers_in']
        self.transfers_in_event = json_data['transfers_in_event']
        self.transfers_out = json_data['transfers_out']
        self.transfers_out_event = json_data['transfers_out_event']
        self.value_form = json_data['value_form']
        self.value_season = json_data['value_season']
        self.web_name = json_data['web_name']
        self.yellow_cards = json_data['yellow_cards']

    @property
    def full_name(self):
        return '{} {}'.format(self.first_name, self.second_name)

    @property
    def position(self):
        return Position(self.element_type)

    def __str__(self):
        return '{:<35}  {:>4} mil, {:>3} points  {} {}'.format(self.full_name, self.now_cost/10, self.total_points,
                                                               self.team, self.position.name)
