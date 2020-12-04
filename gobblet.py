import numpy as np
from itertools import product
from collections import defaultdict
from random import choice
from copy import deepcopy

# reference: a game AI for Connect 4, https://github.com/KeithGalli/Connect4-Python


def get_piece_value(piece):
    """
    Check the value or weight of a piece based on its size

    """
    if piece == 's' or piece == 'S':
        value = 1
    elif piece == 'm' or piece == 'M':
        value = 2
    elif piece == 'l' or piece == 'L':
        value = 3
    elif piece == 'xl' or piece == 'XL':
        value = 4
    else:
        value = 0
    return value


class Board:

    def __init__(self, height=4, width=4):
        """
        :param height: height of the game board (int)
        :param width: width of the game board (int)
        :param board: the game board storing every layer of each cell ({coordinate: [piece1, piece2,...]})
        :param top_layer: the top layer of the game board (np.array)
        :param top_layer_value: the piece value of each cell on the top layer (np.array)
        :param current_turn: used to identify the current player

        """
        self.height = 4
        self.width = 4
        # self.board = {}
        # for coord in list(product(range(height), range(width))):
        #     self.board[coord] = []
        # self.top_layer = np.array([[' ' for _ in range(width)] for _ in range(height)], dtype=object)  # top layer of the game board
        # self.top_layer_value = np.array([[0 for _ in range(width)] for _ in range(height)])
        self.board = None
        self.top_layer = None
        self.top_layer_value = None
        self.reset_board()
        self.current_turn = None

    def reset_board(self):
        """
        reset the game board when game over

        """
        self.board = {}
        for coord in list(product(range(self.height), range(self.width))):
            self.board[coord] = []
        self.top_layer = np.array([[' ' for _ in range(self.width)] for _ in range(self.height)], dtype=object)  # top layer of the game board
        self.top_layer_value = np.array([[0 for _ in range(self.width)] for _ in range(self.height)])

    def get_top_layer(self):
        """
        update the top layer and corresponding piece values of top layer when the board gets updated

        """
        for coord, pieces in self.board.items():
            r, c = coord
            if len(pieces) > 0:
                self.top_layer[r][c] = pieces[-1]
                self.top_layer_value[r][c] = get_piece_value(pieces[-1])
            else:
                self.top_layer[r][c] = ' '
                self.top_layer_value[r][c] = 0

    def game_over(self):
        """
        check if the board game is over and the result for current player
        :return: 'continue': game is not over
                  'draw': a draw
                  'win': the current player wins
                  'lose': the current player loses
        """
        is_upper = np.char.isupper(self.top_layer.astype(str))
        is_lower = np.char.islower(self.top_layer.astype(str))
        upper_in_row = np.hstack((np.sum(is_upper, axis=0), np.sum(is_upper, axis=1),
                                 np.sum(np.diagonal(is_upper)), np.sum(np.diagonal(np.rot90(is_upper)))))
        lower_in_row = np.hstack((np.sum(is_lower, axis=0), np.sum(is_lower, axis=1), np.sum(np.diagonal(is_lower)),
                                 np.sum(np.diagonal(np.rot90(is_lower)))))

        if 4 in upper_in_row and 4 in lower_in_row:
            return 'draw'
        elif 4 in upper_in_row:
            return 'win' if self.current_turn else 'lose'
        elif 4 in lower_in_row:
            return 'lose' if self.current_turn else 'win'
        else:
            return 'continue'

        # the following code is kind of more efficient, but cannot identify a draw
        # row_pieces = list(''.join(r) for r in self.top_layer)
        # for row in row_pieces:
        #     if not row.isalpha():
        #         continue
        #     if row.isupper():
        #         return 'win' if uppercase else 'lose'
        #     elif row.islower():
        #         return 'lose' if uppercase else 'win'
        # column_pieces = list(''.join(self.top_layer[:,c]) for c in range(self.width))
        # for column in column_pieces:
        #     if not column.isalpha():
        #         continue
        #     if column.isupper():
        #         return 'win' if uppercase else 'lose'
        #     elif column.islower():
        #         return 'lose' if uppercase else 'win'
        # diagonal_pieces = [''.join(np.diagonal(self.top_layer)), ''.join(np.diagonal(np.rot90(self.top_layer)))]
        # for diagonal in diagonal_pieces:
        #     if not diagonal.isalpha():
        #         continue
        #     if diagonal.isupper():
        #         return 'win' if uppercase else 'lose'
        #     elif diagonal.islower():
        #         return 'lose' if uppercase else 'win'
        # return 'continue'


class Player:

    def __init__(self, name, uppercase=True, mode='R', piece_amount=3):
        """

        :param name: player's name (str)
        :param uppercase: if the letters used by the player as piece symbols are uppercase (bool)
        :param mode: 'A' measn the AI mode
                     'H' means the human mode with input through console
                     'R' mean the random mode, player will make a move randomly
        :param piece_amount: the amount of piece for each size (int)
        :param pieces: the letters to represent all types of pieces (list)
        :param opponent: the player's opponent (Player)
        :param win_count: how many times the player wins (int)
        :param lose_count: how many times the player loses (int)
        :param draw_count: how many times there are draws (int)
        :param piece_positions: the current positions of each type of piece, (-1,-1) means off-board ({'S':[pos1,pos2...],...})
        :param quit: if the player decide to quit the game

        """
        self.name = name
        self.uppercase = uppercase
        self.piece_amount = piece_amount
        self.mode = mode
        self.opponent = None
        if uppercase:
            self.pieces = ['S', 'M', 'L', 'XL']
        else:
            self.pieces = ['s', 'm', 'l', 'xl']
        self.win_count = 0
        self.lose_count = 0
        self.draw_count = 0
        self.piece_positions = None
        self.reset_player()
        self.quit = False

    def reset_player(self):
        """
        reset the piece positions when game is over

        """
        self.piece_positions = {}
        for piece in self.pieces:
            self.piece_positions[piece] = [(-1, -1) for _ in range(self.piece_amount)]

    def __eq__(self, other):
        return self.name == other.name and self.pieces == other.pieces and self.piece_amount == other.piece_amount

    def find_valid_move(self, board:Board):
        """
        find all possible moves in current turn
        :param board: the game board
        :return: possible moving path for each piece {(start position, target position): ['S', 'M',...],...}
        """
        moves = defaultdict(list)
        for piece in self.pieces:
            current_positions = list(set(self.piece_positions[piece]))
            start_positions = []
            for coord in current_positions:
                if coord == (-1,-1) or board.top_layer[coord[0]][coord[1]] == piece:
                    start_positions.append(coord)
            # print('piece: ', piece, 'start: ', start_positions, ' cur: ', current_positions)
            if len(start_positions) == 0:
                continue
            #print(board.top_layer_value<get_piece_value(piece))
            target_positions = list(map(tuple, np.argwhere(board.top_layer_value<get_piece_value(piece))))
            #print(target_positions)
            if len(target_positions) == 0:
                continue
            possible_moves = list(product(start_positions, target_positions))
            #print(possible_moves)
            for move in possible_moves:
                moves[move].append(piece)
            #print(moves)
        return moves

    def choose_move(self, board:Board):
        """
        choose the move according to the player's mode
        :param board: the game board
        :return: start position, target position, piece to be moved
        """
        def minimax(board:Board, depth, alpha, beta, maximizingPlayer=self.uppercase):
            """
            the player using uppercase letters is maximizing player
            the player using lowercase letters is minimizing player
            """
            if maximizingPlayer == self.uppercase:
                moves = self.find_valid_move(board)
            else:
                moves = self.opponent.find_valid_move(board)
            result = board.game_over()
            if depth == 0 or result != 'continue':
                return None, None, None, self.evaluate_score(board)

            if maximizingPlayer:
                value = -np.inf
                s, t, p = None, None, None
                for move in moves:
                    start, target = move
                    board_copy = deepcopy(board)
                    for piece in moves[move]:
                        if maximizingPlayer == self.uppercase:
                            piece_positions_copy = deepcopy(self.piece_positions)
                            self.drop_piece(start, target, piece, board_copy)
                            turn_value = minimax(board_copy, depth - 1, alpha, beta, False)[3]
                            self.piece_positions = piece_positions_copy
                        else:
                            piece_positions_copy = deepcopy(self.opponent.piece_positions)
                            self.opponent.drop_piece(start, target, piece, board_copy)
                            turn_value = minimax(board_copy, depth-1, alpha, beta, False)[3]
                            self.opponent.piece_positions = piece_positions_copy
                        if turn_value > value:
                            value = turn_value
                            s, t, p = start, target, piece
                            # print(s, t, p, turn_value)
                        alpha = max(alpha, value)
                        if alpha >= beta:
                            break
                    else:
                        continue
                    break
                return s, t, p, value
            else:
                value = np.inf
                s, t, p = None, None, None
                for move in moves:
                    start, target = move
                    board_copy = deepcopy(board)
                    for piece in moves[move]:
                        if maximizingPlayer == self.uppercase:
                            piece_positions_copy = deepcopy(self.piece_positions)
                            self.drop_piece(start, target, piece, board_copy)
                            turn_value = minimax(board_copy, depth - 1, alpha, beta, True)[3]
                            self.piece_positions = piece_positions_copy
                        else:
                            piece_positions_copy = deepcopy(self.opponent.piece_positions)
                            self.opponent.drop_piece(start, target, piece, board_copy)
                            turn_value = minimax(board_copy, depth - 1, alpha, beta, True)[3]
                            self.opponent.piece_positions = piece_positions_copy
                        if turn_value < value:
                            value = turn_value
                            s, t, p = start, target, piece
                        beta = min(beta, value)
                        if alpha >= beta:
                            break
                    else:
                        continue
                    break
                return s, t, p, value
        # AI mode
        if self.mode == 'A':
            start, target, piece, score = minimax(board, 2, -np.inf, np.inf)
            # print('score', score)
        # human mode
        elif self.mode == 'H':
            human_move = input(
                'Please type your move(start point, target point, piece), ex. (0,0),(1,0),\'s\', or type Q to quit: ')  # format:(0,0),(1,0)
            if human_move == 'Q':
                self.quit = True
                start, target, piece = None, None, None
            else:
                start, target, piece = (eval(human_move))
        # randomness mode
        else:
            moves = self.find_valid_move(board)
            move = choice(list(moves.keys()))
            piece = choice(moves[move])
            start, target = move
        return start, target, piece


    def drop_piece(self, start, target, piece, board:Board):
        """
        move the piece from the start position to the target position on the board

        """
        if start != (-1,-1):
            board.board[start].pop()
        board.board[target].append(piece)
        # print(self.piece_positions)
        # print(piece)
        self.piece_positions[piece].remove(start)
        self.piece_positions[piece].append(target)
        board.get_top_layer()
        return

    def evaluate_score(self, board: Board):
        """
        heuristic function to calculate the value of a given board pattern

        """

        def score_row(row, row_indicator):
            score = 0
            num_oppo = np.count_nonzero(row_indicator == -1)
            num_blank = np.count_nonzero(row_indicator == 0)
            if num_oppo == 4:
                score -= 1000
            elif num_oppo == 3:
                if num_blank == 1:
                    score -= 50
                elif num_blank == 0:
                    my_index = (row_indicator == 1).nonzero()[0]
                    my_piece = row[my_index]
                    my_value = get_piece_value(my_piece)
                    if my_value < 4:
                        score = score + 5*(my_value - 10)
            elif num_oppo == 2:
                if num_blank == 0:
                    for piece in row:
                        score -= get_piece_value(piece) # not sure
            elif num_oppo == 0:
                if num_blank == 2:
                    for piece in row:
                        score += 5*get_piece_value(piece)
                if num_blank == 1:
                    score += 50
                    for piece in row:
                        score += 5*get_piece_value(piece)
                if num_blank == 0:
                    score += 1000
            return score

        score = 0
        top_layer = board.top_layer.astype(str)
        key_positions = [
            (0, 0), (0, 3), (3, 0), (3, 3),
            (1, 1), (1, 2), (2, 1), (2, 2)
        ]
        for pos in key_positions:
            r, c = pos
            piece = board.top_layer[r][c]
            if piece in self.pieces:
                #score = score + get_piece_value(piece)
                score += 2

        if self.uppercase:
            indicator = np.where(np.char.isupper(top_layer), 1, np.where(np.char.islower(top_layer), -1, 0))
        else:
            indicator = np.where(np.char.islower(top_layer), 1, np.where(np.char.isupper(top_layer), -1, 0))

        for c in range(board.width):
            row = top_layer[:, c]
            row_indicator = indicator[:, c]
            score += score_row(row, row_indicator)

        for r in range(board.height):
            row = top_layer[r, :]
            row_indicator = indicator[r, :]
            score += score_row(row, row_indicator)

        diag = np.diagonal(top_layer)
        diag_indicator = np.diagonal(indicator)
        score += score_row(diag, diag_indicator)

        back_diag = np.diagonal(np.rot90(top_layer))
        back_diag_indicator = np.diagonal(np.rot90(indicator))
        score += score_row(back_diag, back_diag_indicator)

        return score if self.uppercase else -1*score


def tournament(p1:Player, p2:Player, board:Board, times=100, verbose=False):
    """
    competition between two players
    :param p1: player1
    :param p2: player2
    :param board: game board
    :param times: number of games
    :param verbose: if print the details of the competition
    :return: the final score for each player
    """
    turns = 0
    p1.opponent = p2
    p2.opponent = p1
    while turns < times:
        game_over = False
        board.current_turn = choice([0,1])
        #current_turn = choice([0,1])
        #current_turn = 1
        while not game_over:
            current_player = p1 if p1.uppercase == board.current_turn else p2
            opponent = p2 if current_player == p1 else p1
            # moves = current_player.find_valid_move(board)
            # start, target, piece = current_player.choose_move(moves)
            # print(current_player.find_valid_move(board))
            start, target, piece = current_player.choose_move(board)
            if current_player.quit:
                exit()
            current_player.drop_piece(start, target, piece, board)
            #result = board.game_over(current_player.uppercase)
            result = board.game_over()
            if verbose:
                print(current_player.name, 'move: ', start, target, piece)
                print('The whole board:\n', board.board)
                print('The top layer of the board:\n',board.top_layer)
                #print(board.top_layer_value)
            if result == 'continue':
                board.current_turn ^= 1
            else:
                if result == 'win':
                    current_player.win_count += 1
                    opponent.lose_count += 1
                elif result == 'draw':
                    current_player.draw_count += 1
                    opponent.draw_count += 1
                else:
                    current_player.lose_count += 1
                    opponent.win_count += 1
                game_over = True


        print('Final board','\n',board.top_layer)
        print('Result: ', current_player.name, ' ', result)
        #print(p1.win_count, p2.win_count)

        turns += 1
        #print(p2.piece_positions)
        board.reset_board()
        p1.reset_player()
        p2.reset_player()

    print('Out of {} games, {} wins {} times, {} wins {} times'.format(times, p1.name, p1.win_count, p2.name, p2.win_count))
    return

board = Board(4, 4)
p1 = Player('p1', True, mode='R')
p2 = Player('p2', False, mode='R')
tournament(p1, p2, board, 100, verbose=False)

