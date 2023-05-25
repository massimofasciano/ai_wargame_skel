import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 1000000
MIN_HEURISTIC_SCORE = -1000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> 'Player':
        """The other player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Defender

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
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: 'Unit') -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: 'Unit') -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> 'Coord':
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable['Coord']:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable['Coord']:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> 'Coord | None':
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> 'CoordPair':
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> 'CoordPair':
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> 'CoordPair':
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> 'CoordPair|None':
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
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
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    randomize_moves : bool = True

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0

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

    def clone(self) -> 'Game':
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit|None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit|None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and unit.health <= 0:
            self.set(coord,None)

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            if not target.is_alive():
                self.set(coord,None)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        return self.validate_and_move(coords,False)

    def move_unit(self, coords : CoordPair) -> bool:
        """Validate and perform a move expressed as a CoordPair."""
        return self.validate_and_move(coords,True)

    def check_move_range(self, coords: CoordPair) -> bool:
        """Is this move within the movement range of a unit?"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        if unit is None:
            return False
        elif unit.type is UnitType.Tech or unit.type is UnitType.Virus:
            # tech and virus can retreat
            return (
                abs(coords.dst.col-coords.src.col)+
                abs(coords.dst.row-coords.src.row)
            ) == 1
        elif unit.player is Player.Attacker:
            # attacker moves top and left
            return (
                coords.src.col-coords.dst.col+
                coords.src.row-coords.dst.row
            ) == 1
        else:
            # defender moves bottom and right
            return (
                coords.dst.col-coords.src.col+
                coords.dst.row-coords.src.row
            ) == 1

    def is_engaged(self, src: Coord) -> bool:
        """Is the unit at Coord engaged in combat with opposing units ?"""
        source = self.get(src)
        if source is None:
            return False
        for dst in src.iter_adjacent():
            target = self.get(dst)
            if target is not None and target.player != source.player:
                return True
        return False

    def validate_and_move(self, coords : CoordPair, perform_move: bool) -> bool:
        """Validate and optionally perform a move expressed as a CoordPair."""
        source = self.get(coords.src)
        if source is None or not self.is_valid_coord(coords.dst):
            return False
        target = self.get(coords.dst)
        if source.player == self.next_player:
            if coords.src == coords.dst:
                # we self destruct (side effects missing!!!)
                if perform_move:
                    for coord in coords.src.iter_range(1):
                        unit = self.get(coord)
                        if unit is not None:
                            unit.mod_health(-2)
                            self.remove_dead(coord)
                    self.set(coords.src,None)
                return True
            elif target is None and self.check_move_range(coords) and not self.is_engaged(coords.src):
                # we move the unit!
                if perform_move:
                    self.set(coords.dst,source)
                    self.set(coords.src,None) 
                return True
            elif target is not None and target.player != source.player and self.check_move_range(coords):
                # we attack opposing unit!
                if perform_move:
                    target.mod_health(-source.damage_amount(target))
                    source.mod_health(-target.damage_amount(source))
                    self.remove_dead(coords.src)
                    self.remove_dead(coords.dst)
                return True
            elif target is not None and target.player == source.player and self.check_move_range(coords):
                # we repair friendly unit
                amount = source.repair_amount(target)
                if amount < 1:
                    # not valid move if repair amount is 0
                    return False
                if perform_move:
                    target.mod_health(amount)
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
            label = coord.col_string()
            output += f"{label:^5}"
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
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
        """Human player plays a move."""
        self.get_move()
        while True:
            mv = self.read_move()
            if self.move_unit(mv):
                self.next_turn()
                break
            else:
                print("The move is not valid! Try again.")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        move = self.suggest_move()
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
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
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
                return None
        if attacker_has_ai:
            return Player.Attacker
        return Player.Defender

    def apply_heuristic(self, player: Player, maximizing: bool, depth: int, winner: Player | None) -> int:
        """Apply custom heuristic evaluation after some general calculations. Potential winner needs to be precalculated."""
        if depth not in self.stats.evaluations_per_depth:
            self.stats.evaluations_per_depth[depth] = 1
        else:
            self.stats.evaluations_per_depth[depth] += 1
        # we could use "maximizing" to select different heuristics for min and max stages
        if winner is None:
            return self.heuristic_e2(player)
        elif winner == player:
            # prefer to win earlier
            return MAX_HEURISTIC_SCORE - self.turns_played
        else:
            # prefer to lose later
            return MIN_HEURISTIC_SCORE + self.turns_played

    def heuristic_e1(self, player: Player):
        """Heuristic based on score per unit type."""
        def get_score(cu : Tuple[Coord,Unit]) -> int:
            unit = cu[1]
            score = 1
            if unit.type == UnitType.Tech or unit.type == UnitType.Virus:
                # better units
                score = 3
            if unit.type == UnitType.AI:
                # already lose or win
                score = 0
            return score
        return (
            sum(map(get_score,self.player_units(player))) -
            sum(map(get_score,self.player_units(player.next())))
        )

    def heuristic_e2(self, player: Player):
        """Heuristic based on unit health and score per unit type and game turns."""
        def get_score(cu : Tuple[Coord,Unit]) -> int:
            unit = cu[1]
            score = 1
            if unit.type == UnitType.Tech or unit.type == UnitType.Virus:
                # better units
                score = 3
            if unit.type == UnitType.AI:
                # already lose or win
                score = 0
            return unit.health+100*score
        return (
            sum(map(get_score,self.player_units(player))) -
            sum(map(get_score,self.player_units(player.next()))) +
            10 * self.turns_played
        )

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_range(1):
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()

    def suggest_move(self) -> CoordPair|None:
        """Suggest the next move using minimax alpha beta."""
        start_time = datetime.now()
        (score, move, avg_depth) = self.minimax_alpha_beta(True, self.next_player, 0, MIN_HEURISTIC_SCORE, MAX_HEURISTIC_SCORE, start_time)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Evals per depth: ",end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        return move

    def minimax_alpha_beta(self, maximizing: bool, player: Player, depth: int, alpha: int, beta: int, start_time: datetime) -> Tuple[int, CoordPair|None, float]:
        """Minimax with alpha beta pruning."""
        time_now = datetime.now()
        elapsed_seconds = (time_now - start_time).total_seconds()
        timeout = False
        if self.options.max_time is not None and elapsed_seconds > self.options.max_time:
            timeout = True
        winner = self.has_winner()
        if ((timeout and self.options.min_depth is not None and depth >= self.options.min_depth) or
           (self.options.max_depth is not None and depth >= self.options.max_depth) or
           winner is not None):
            return (self.apply_heuristic(player,maximizing,depth,winner),None,depth)
        else:
            best_move = None
            if maximizing:
                best_score = MIN_HEURISTIC_SCORE
            else:
                best_score = MAX_HEURISTIC_SCORE
            total_depth = 0.0
            total_count = 0
            move_candidates = self.move_candidates()
            if self.options.randomize_moves:
                # turn iterator into list
                move_candidates = list(move_candidates)
                # and shuffle it randomly
                random.shuffle(move_candidates)
            for move_candidate in move_candidates:
                new_game_state = self.clone()
                if not new_game_state.move_unit(move_candidate):
                    continue
                new_game_state.next_turn()
                (score, _, rec_avg_depth) = new_game_state.minimax_alpha_beta(not maximizing, player, depth+1, alpha, beta, start_time)
                total_depth += rec_avg_depth
                total_count += 1
                if (maximizing and score >= best_score) or (not maximizing and score <= best_score):
                    best_score = score
                    best_move = move_candidate
                if self.options.alpha_beta:
                    if maximizing:
                        if best_score > beta:
                            break
                        alpha = max(alpha, best_score)
                    else:
                        if best_score < alpha:
                            break
                        beta = min(beta, best_score)
            if total_count == 0:
                return (self.apply_heuristic(player,maximizing,depth,winner),None,depth)
            else:
                return (best_score, best_move, total_depth / total_count)

    def post_move(self, move: CoordPair):
        if broker_url is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(broker_url, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move(self) -> CoordPair | None:
        if broker_url is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(broker_url, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        print("Got broker data for wrong turn")
                        pass
                else:
                    print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

broker_url = "http://192.168.140.40:8001/test"
# broker_url = None

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--game_type', type=str, default="auto", help='game type: auto|attacker|defender|manual')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time

    # create a new game
    game = Game(options=options)

    # the main game loop
    while True:
        print()
        print(game)
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            move = game.computer_turn()
            if move is not None:
                game.post_move(move)
                print(f"Computer played {move}")
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################

if __name__ == '__main__':
    main()
