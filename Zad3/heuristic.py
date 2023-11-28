from typing import Tuple


def heuristic(fields, maximizing_player) -> int:
    score = 0
    exp_heur = {0:0, 1: 1, 2: 5, 3: 20, 4: 1000}
    for column_id in range(len(fields)):  # verticals
        for start_row_id in range(len(fields[column_id]) - 3):
            points_dict = count_lines(fields, (column_id, start_row_id), (0, 1))
            if len(points_dict) == 1:
                if maximizing_player.char in points_dict:
                    series = points_dict[maximizing_player.char]
                    score += exp_heur[series]
                else:
                    series = list(points_dict.values())
                    score -= exp_heur[series[0]]
    for start_column_id in range(len(fields) - 3):  # horizontals
        for row_id in range(len(fields[start_column_id])):
            points_dict = count_lines(fields, (start_column_id, row_id), (1, 0))
            if len(points_dict) == 1:
                if maximizing_player.char in points_dict:
                    series = points_dict[maximizing_player.char]
                    score += exp_heur[series]
                else:
                    series = list(points_dict.values())
                    score -= exp_heur[series[0]]
    for start_column_id in range(len(fields) - 3):  # diagonals
        for start_row_id in range(len(fields[start_column_id]) - 3):
            points_dict = count_lines(fields, (start_column_id, start_row_id), (1, 1))
            if len(points_dict) == 1:
                if maximizing_player.char in points_dict:
                    series = points_dict[maximizing_player.char]
                    score += exp_heur[series]
                else:
                    series = list(points_dict.values())
                    score -= exp_heur[series[0]]
            points_dict = count_lines(fields, (start_column_id, start_row_id + 3), (1, -1))
            if len(points_dict) == 1:
                if maximizing_player.char in points_dict:
                    series = points_dict[maximizing_player.char]
                    score += exp_heur[series]
                else:
                    series = list(points_dict.values())
                    score -= exp_heur[series[0]]
    return score


def count_lines(fields, start_coords: Tuple[int, int], move_coords: Tuple[int, int]) -> dict:
    points = {}
    prev_field_value = fields[start_coords[0]][start_coords[1]]
    for i in range(4):
        row_index = start_coords[0] + move_coords[0] * i
        col_index = start_coords[1] + move_coords[1] * i
        field_value = fields[row_index][col_index]
        if field_value is not None:
            if prev_field_value is None:
                points[field_value.char] = points.get(field_value.char, 0) + 1
            elif prev_field_value.char == field_value.char:
                points[field_value.char] = points.get(field_value.char, 0) + 1
            else:
                points[field_value.char] = 0
                points[prev_field_value.char] = 0
                break
    return points
