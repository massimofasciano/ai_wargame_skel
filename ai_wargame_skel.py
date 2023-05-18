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
