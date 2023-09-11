This is a skeleton Python program implementing AI Wargame.

It is given to you to help bootstrap your project and save some time.
Some parts have been removed from the complete code. 
You will have to write those parts. Make sure all of the rules of the game are properly implemented and don't assume that this code checks everything.

The code uses a lot of abstractions for types and makes ample use of iterators. This was done to make it clearer and is not necessarily what gives the best performance. If you want to try to make the code faster, feel free to experiment at this level but remember that Python is a very slow language and this game has a high branching factor and a lot of turns so a performance gain of 2 or 3 will not allow you to explore a lot more search space.

As a quick reference, the Python code is roughly 100x slower than the same program written in a fast compiled language. Using the pypy runtime instead of the regular python runtime will help (expect at least a 2x speedup).

The code uses classes to represent the following elements:

UnitType(Enum): the various types of Units
Player(Enum): the 2 players
GameType(Enum): various game types (player vs comp, etc..)
Unit: a Unit (belongs to a player, has a type and health)
Coord: a game coordinate on the board
CoordPair: a pair of game coordinates (represents a move or a rectangular area)
Options: game options
Stats: game statistics
Game: the game state including the board

It also uses iterators for many concepts:
- all units belonging to a player
- adjacent coordinates
- all coordinates in rectangular area of the board
- all possible valid moves from a given game state
- etc...

The program takes command line options to adjust some options. Feel free to add any other options you need.

```
$ python ai_wargame_skel.py -h
usage: ai_wargame [-h] [--max_depth MAX_DEPTH] [--max_time MAX_TIME] [--game_type GAME_TYPE] [--broker BROKER]

options:
  -h, --help            show this help message and exit
  --max_depth MAX_DEPTH
                        maximum search depth (default: None)
  --max_time MAX_TIME   maximum search time (default: None)
  --game_type GAME_TYPE
                        game type: auto|attacker|defender|manual (default: auto)
  --broker BROKER       play via a game broker (default: None)
```

It supports playing via a very simple game broker also written in Python. The broker is a simple network server that stores moves and allows two programs to play against each other without having to enter the moves manually.

```
$ python game_broker.py
serving at http://192.168.1.100:8001/1kcxxmaL
```

You run an instance of the broker on a computer that is not firewalled and it gives you a connection URL.

You then give that URL to the game and set one program as the attacker and the other one as the defender.

```
python ai_wargame_skel.py --game_type defender --broker http://192.168.1.100:8001/1kcxxmaL
```

```
python ai_wargame_skel.py --game_type attacker --broker http://192.168.1.100:8001/1kcxxmaL
```

The two programs will play against each other and post their moves to the broker.

You can also test locally without a broker.
Run the program in auto mode to get the computer to play against itself.

```
$ python ai_wargame_skel.py --game_type auto

Next player: Attacker
Turns played: 0

    0   1   2   3   4
A: dA9 dT9 dF9  .   .
B: dT9 dP9  .   .   .
C: dF9  .   .   .  aP9
D:  .   .   .  aF9 aV9
E:  .   .  aP9 aV9 aA9

Heuristic score: -40
Average recursive depth: 3.4
Evals per depth: 1:1 2:8 3:46 4:4185
Eval perf.: 5.1k/s
Elapsed time: 0.8s
Computer Attacker: D3 moves to D2

Next player: Defender
Turns played: 1

    0   1   2   3   4
A: dA9 dT9 dF9  .   .
B: dT9 dP9  .   .   .
C: dF9  .   .   .  aP9
D:  .   .  aF9  .  aV9
E:  .   .  aP9 aV9 aA9

etc...
```
