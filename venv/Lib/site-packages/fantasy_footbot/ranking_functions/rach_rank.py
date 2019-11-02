from fantasy_footbot.entities.player import Player


def rach_rank(player: Player, prev_season_name="2017/18", games_per_season=38):
    print(player)
    # undesirable statuses are s, i, d - suspended, injured, d... dead???
    # a = available
    if player.status != 'a':
        return 0.0

    # ignore any player active less than 60 mins per game this season
    if player.minutes <= len(player.history) * 60:
        return 0.0

    for season in player.history_past:
        if season['season_name'] == prev_season_name:
            last_season_ppg = season['total_points'] / games_per_season
            break
    else:
        last_season_ppg = 0

    # player average points per game, as long as player has seen sufficient play
    # if early weeks in season, use last year's data in weighted function
    if len(player.history) > 0:
        current_season_ppg = player.total_points / len(player.history)
    else:
        current_season_ppg = 0

    if len(player.history) < 5:
        # use a mix of last season's performance data to reduce variance from lack of info
        r_rank = current_season_ppg * (len(player.history) / 5) + last_season_ppg * (5 - len(player.history) / 5)
    else:
        r_rank = current_season_ppg
    return r_rank
