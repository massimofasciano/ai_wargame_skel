from enum import Enum
from dataclasses import dataclass
   
class UnitType(Enum):
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    Attacker = 0
    Defender = 1

##############################################################################################################

@dataclass
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9

    def is_alive(self) -> bool:
        return self.health > 0

    def mod_health(self, health_delta : int):
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def string3(self):
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

##############################################################################################################

@dataclass(init=False)
class Game:
    board: list[list[Unit|None]]
    next_player: Player = Player.Attacker
    dim: int = 5
    turns_played : int = 0

    def __init__(self, dim: int|None = None):
         if dim is not None:
             self.dim = dim
         self.board = [[None for _ in range(self.dim)] for _ in range(self.dim)]

    def is_empty(self, row : int, col : int) -> bool:
        return self.board[row][col] is None

    def get(self, row : int, col : int) -> Unit|None:
        return self.board[row][col]

    def set(self, row : int, col : int, unit : Unit|None):
        self.board[row][col] = unit

    def mod_health(self, row : int, col : int, health_delta : int):
        target = self.get(row,col)
        if target is not None:
            target.mod_health(health_delta)
            if not target.is_alive():
                self.set(row,col,None)

    def move_unit(self, from_row : int, from_col: int, to_row : int, to_col: int) -> bool:
        # TODO: must check all other move conditions!
        source = self.get(from_row,from_col)
        if source is not None and source.player == self.next_player and self.is_empty(to_row,to_col):
            self.set(to_row,to_col,source)
            self.set(from_row,from_col,None) 
            return True
        return False

    def next_turn(self):
        if self.next_player == Player.Attacker:
            self.next_player = Player.Defender
        else:
            self.next_player = Player.Attacker
        self.turns_played += 1

    def board_string(self):
        output = ""
        for row in range(self.dim):
            for col in range(self.dim):
                unit = self.get(row,col)
                if unit is None:
                    output += " . "
                else:
                    output += unit.string3()
            output += "\n"
        return output

    def pretty_print(self):
        print(f"Next player: {self.next_player.name}")
        print(f"Turns played: {self.turns_played}")
        print(self.board_string())

    def is_valid(self, row, col):
        if row < 0 or row >= self.dim or col < 0 or col >= self.dim:
            return False
        return True

    def input_move_string(self):
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coord = Coord()
            coord.from_string(s)
            if self.is_valid(coord.row, coord.col):
                return coord
            else:
                print('The move is not valid! Try again.')

##############################################################################################################

@dataclass
class Coord:
    row : int = 0
    col : int = 0

    def col_string(self):
            coord_char = '?'
            if self.col < 16:
                    coord_char = "0123456789abcdef"[self.col]
            return str(coord_char)

    def row_string(self):
            coord_char = '?'
            if self.row < 26:
                    coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
            return str(coord_char)

    def to_string(self):
            return self.row_string()+self.col_string()

    def from_string(self, s : str):
            s = s.strip()
            for sep in " ,.:;-_":
                    s = s.replace(sep, "")
            if (len(s) == 2):
                self.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
                self.col = "0123456789abcdef".find(s[1:2].lower())

##############################################################################################################

g = Game()

g.set(3,4,Unit())
g.set(2,1,Unit(player=Player.Defender, type=UnitType.Virus, health=3))

print(g)
print(g.is_empty(3,4))
print(g.is_empty(4,4))

g.pretty_print()

g.set(3,4,None)
print(g.is_empty(3,4))
print(g.get(3,4))
print(g.get(2,1))

g.pretty_print()

g.mod_health(2,1,-2)
print(g.get(2,1))
g.mod_health(2,1,-2)
print(g.get(2,1))

g.pretty_print()

g.set(2,1,Unit(player=Player.Defender))
print(g.move_unit(2,1,3,2))
g.next_turn()
print(g.move_unit(2,1,3,2))
print(g)

g.pretty_print()

print(g.input_move_string())
