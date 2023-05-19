import argparse
import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, TypeVar, Type, Iterable
import random

class UnitType(Enum):
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    Attacker = 0
    Defender = 1

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

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

    def to_string(self):
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self):
        return self.to_string()

##############################################################################################################

CoordType = TypeVar('CoordType', bound='Coord')

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
    
    def __str__(self):
        return self.to_string()
    
    def clone(self):
        return copy.copy(self)

    @classmethod
    def from_string(cls : Type[CoordType], s : str) -> CoordType|None:
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = cls()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

CoordPairType = TypeVar('CoordPairType', bound='CoordPair')

@dataclass()
class CoordPair:
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self):
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self):
        return self.to_string()

    @classmethod
    def from_string(cls : Type[CoordPairType], s : str) -> CoordPairType|None:
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = cls()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass()
class Options:
    dim: int = 5
    max_depth : int | None = None
    max_time : float | None = None
    game_type : GameType = GameType.AttackerVsDefender

##############################################################################################################

@dataclass()
class Stats:
    total_evaluations : int = 0

##############################################################################################################

@dataclass()
class Game:
    board: list[list[Unit|None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
         dim = self.options.dim
         self.board = [[None for _ in range(dim)] for _ in range(dim)]
         md = dim-1
         self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
         self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
         self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
         self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Program))
         self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Program))
         self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Firewall))
         self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
         self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
         self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
         self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Firewall))
         self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Firewall))
         self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Program))

    def clone(self):
        # make a shallow copy of everything except the board (options and stats are shared)
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit|None:
        return self.board[coord.row][coord.col]

    def set(self, coord : Coord, unit : Unit|None):
        self.board[coord.row][coord.col] = unit

    def mod_health(self, coord : Coord, health_delta : int):
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            if not target.is_alive():
                self.set(coord,None)

    def move_unit(self, coords : CoordPair) -> bool:
        # TODO: must check all other move conditions!
        source = self.get(coords.src)
        if source is not None and source.player == self.next_player:
            if self.is_empty(coords.dst):
                # we move it (many checks missing!!!)
                self.set(coords.dst,source)
                self.set(coords.src,None) 
                return True
            elif coords.src == coords.dst:
                # we self destruct (side effects missing!!!)
                self.set(coords.src,None)
                return True
        return False

    def next_turn(self):
        if self.next_player == Player.Attacker:
            self.next_player = Player.Defender
        else:
            self.next_player = Player.Attacker
        self.turns_played += 1

    def to_string(self):
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "   "
        for col in range(dim):
            coord.col = col
            label = coord.to_string()[1]
            output += f"{label:^5}"
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.to_string()[0]
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += "  .  "
                else:
                    output += f"{str(unit):^5}"
            output += "\n"
        return output

    def __str__(self):
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord):
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self):
        while True:
            mv = self.read_move()
            if self.move_unit(mv):
                self.next_turn()
                break
            else:
                print("The move is not valid! Try again.")

    def computer_turn(self):
        dim = self.options.dim
        while True:
            d1 = random.randint(0, dim-1)
            d2 = random.randint(0, dim-1)
            d3 = random.randint(0, dim-1)
            d4 = random.randint(0, dim-1)
            mv = CoordPair(Coord(d1,d2),Coord(d3,d4))
            if self.move_unit(mv):
                self.next_turn()
                break

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]] :
        dim = self.options.dim
        coord = Coord()
        for row in range(dim):
            coord.row = row
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is not None and unit.player == player:
                    yield (coord.clone(),unit)

    def is_finished(self) -> bool:
        attacker_has_ai = False
        defender_has_ai = False
        dim = self.options.dim
        coord = Coord()
        for row in range(dim):
            coord.row = row
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is not None:
                    if unit.player == Player.Attacker and unit.type == UnitType.AI:
                        attacker_has_ai = True
                    if unit.player == Player.Defender and unit.type == UnitType.AI:
                        defender_has_ai = True
                if attacker_has_ai and defender_has_ai:
                    print("both ai")
                    return False
        return True
    
##############################################################################################################

def just_testing():
    g = Game()

    d4 = Coord.from_string("D4")
    assert(d4 is not None)
    c1 = Coord(2,1)

    g.set(d4,Unit())
    g.set(c1,Unit(player=Player.Defender, type=UnitType.Virus, health=3))

    print(repr(g))
    print(g)
    print(g.is_empty(d4))
    print(g.is_empty(Coord(4,4)))

    g.set(d4,None)
    print(g.is_empty(d4))
    print(g.get(d4))
    print(f"{g.get(c1)!r}")
    print(g.get(c1))
    print(repr(g.get(c1)))


    g.mod_health(c1,-2)
    print(g.get(c1))
    g.mod_health(c1,-2)
    print(g.get(c1))


    g.set(c1,Unit(player=Player.Defender))
    print(g)
    mv2132 = CoordPair(c1,Coord(3,2))
    print(g.move_unit(mv2132))
    g.next_turn()
    print(g.move_unit(mv2132))
    print(g)

    g.human_turn()
    print(g)
    print(repr(g))

##############################################################################################################

def main():
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="auto", help='game type: auto|attacker|defender')
    args = parser.parse_args()
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    else:
        game_type = GameType.CompVsComp
    options = Options(max_depth=args.max_depth, max_time=args.max_time, game_type=game_type)
    game = Game(options=options)
    print(repr(game))

    while True:
        print(game)
        game.human_turn()
        if game.is_finished():
            print("Game over!")
            break
        print(game)
        game.computer_turn()
        if game.is_finished():
            print("Game over!")
            break

if __name__ == '__main__':
    # just_testing()
    main()
