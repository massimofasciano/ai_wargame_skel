import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, TypeVar, Type, Iterable, Self
import random

MAX_HEURISTIC_SCORE = 1000000
MIN_HEURISTIC_SCORE = -1000000

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

@dataclass(slots=True)
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

    def to_string(self) -> str:
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        return self.to_string()

##############################################################################################################

CoordType = TypeVar('CoordType', bound='Coord')

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        return self.to_string()
    
    def clone(self) -> Self:
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Self]:
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

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

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        return self.to_string()

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls : Type[CoordPairType], row0: int, col0: int, row1: int, col1: int) -> CoordPairType:
        return cls(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls : Type[CoordPairType], dim: int) -> CoordPairType:
        return cls(Coord(0,0),Coord(dim-1,dim-1))
    
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

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = None
    min_depth : int | None = None
    max_time : float | None = None
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    total_evaluations : int = 0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit|None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
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

    def clone(self) -> Self:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit|None:
        """Get contents of a board cell of the game at Coord."""
        return self.board[coord.row][coord.col]

    def set(self, coord : Coord, unit : Unit|None):
        """Set contents of a board cell of the game at Coord."""
        self.board[coord.row][coord.col] = unit

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            if not target.is_alive():
                self.set(coord,None)

    def move_unit(self, coords : CoordPair) -> bool:
        """Perform a move expressed as a CoordPair."""
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
        """Transitions game to the next turn."""
        if self.next_player == Player.Attacker:
            self.next_player = Player.Defender
        else:
            self.next_player = Player.Attacker
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
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

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
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

    def computer_turn(self) -> CoordPair | None:
        move = self.find_move()
        if move is not None and self.move_unit(move):
            self.next_turn()
        return move

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        attacker_has_ai = False
        defender_has_ai = False
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None:
                if unit.player == Player.Attacker and unit.type == UnitType.AI:
                    attacker_has_ai = True
                if unit.player == Player.Defender and unit.type == UnitType.AI:
                    defender_has_ai = True
            if attacker_has_ai and defender_has_ai:
                return False
        return True

    def heuristic(self,player: Player,maximizing_player: bool, depth: int) -> int:
        self.stats.total_evaluations += 1
        return random.randint(MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE)

    def move_candidates(self) -> Iterable[CoordPair]:
        for (src,_) in self.player_units(self.next_player):
            for dst in CoordPair.from_dim(self.options.dim).iter_rectangle():
                yield CoordPair(src,dst)

    def find_move(self) -> CoordPair|None:
        start_time = datetime.now()
        (score, move, avg_depth) = self.minimax_alpha_beta(True, self.next_player, 0, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE, start_time)
        return move

    def minimax_alpha_beta(self, maximizing_player: bool, player: Player, depth: int, alpha: int, beta: int, start_time) -> Tuple[int, CoordPair|None, float]:
        time_now = datetime.now()
        elapsed_seconds = time_now - start_time
        timeout = False
        if self.options.max_time is not None and elapsed_seconds > self.options.max_time:
            timeout = True 
        if ((timeout and self.options.min_depth is not None and depth >= self.options.min_depth) or
           (self.options.max_depth is not None and depth >= self.options.max_depth) or
           self.is_finished()):
            return (self.heuristic(player,maximizing_player,depth),None,depth)
        else:
            best_move = None
            if maximizing_player:
                best_score = MIN_HEURISTIC_SCORE
            else:
                best_score = MAX_HEURISTIC_SCORE
            total_depth = 0.0
            total_count = 0
            for move_candidate in self.move_candidates():
                new_game_state = self.clone()
                if not new_game_state.move_unit(move_candidate):
                    continue
                (score, _, rec_avg_depth) = new_game_state.minimax_alpha_beta(not maximizing_player, player, depth+1, alpha, beta, start_time)
                total_depth += rec_avg_depth
                total_count += 1
                if (maximizing_player and score >= best_score) or (not maximizing_player and score <= best_score):
                    best_score = score
                    best_move = move_candidate
                if self.options.alpha_beta:
                    if maximizing_player:
                        if best_score > beta:
                            break
                        alpha = max(alpha, best_score)
                    else:
                        if best_score < alpha:
                            break
                        beta = min(beta, best_score)
            if total_count == 0:
                return (self.heuristic(player,maximizing_player,depth),None,depth)
            else:
                return (best_score, best_move, total_depth / total_count)

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

def just_testing2():
    cp = CoordPair.from_quad(2,3,5,6)
    print(cp)
    print()    
    for c in cp.iter_rectangle():
        print(c)
    c = Coord(4,5)

    print()

    print(c)
    print()
    for c in c.iter_range(1):
        print(c)

    g = Game()
    for (c,u) in g.player_units(Player.Defender):
        print(f"{c} => {u}")

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
        move = game.computer_turn()
        print(f"Computer played {move}")
        if move is None or game.is_finished():
            print("Game over!")
            break

if __name__ == '__main__':
    # just_testing2()
    main()
