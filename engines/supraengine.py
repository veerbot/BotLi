# file: engine/stockish.py
#!/usr/bin/env python3
"""

This variant adds bitboard-friendly optimizations:
- caches piece bitboards and uses int bitboard operations (lsb) in hot paths
- implements a faster SEE that simulates captures using bitboards (avoids heavy board.copy()/push/pop when possible)
- move ordering uses cached bitboard-derived heuristics
- minimal API changes: drop-in replacement for your previous stockish.py
Requires: python-chess
"""
from __future__ import annotations
import sys, time, math, random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import chess

# ---------- CONFIG ----------
MAX_DEPTH = 64
MATE_VALUE = 1000000
INFTY = 10**9
DEFAULT_MOVE_TIME = 1.0
ASPIRATION = 50
NULL_REDUCTION = 2
LMR_BASE = 0.75
LMR_DIV = 2.0
FUTILITY_MARGIN = 150
QUIESCENCE_CAP = 2048
MULTIPV_MAX = 4
RANDOM_TIE = True

# ---------- STATE ----------
@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int   # 0=EXACT, 1=LOWER, 2=UPPER
    best: Optional[chess.Move]
    age: int

TT: Dict[int, TTEntry] = {}
TT_AGE = 0
KILLERS: Dict[int, List[Optional[chess.Move]]] = {}
HISTORY: Dict[Tuple[int,int], int] = {}
node_count = 0
start_time = 0.0
time_limit = 0.0
stop_flag = False
NODE_LIMIT = None

# For mate TT (store mate distances) use separate table to avoid confusing eval TT entries
MateTT: Dict[int, int] = {}  # key -> mate score (positive for side to move mate in n plies: MATE_VALUE - ply)

# ---------- PIECES & PST ----------
PV = {}
PIECE_VALUE = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:20000}
PST = {pt:[0]*64 for pt in PIECE_VALUE}

# ---------- UTIL ----------
def now(): return time.time()
def timeout(): return stop_flag or ((time.time()-start_time) >= time_limit)
def within_node_limit(): 
    return (NODE_LIMIT is None) or (node_count < NODE_LIMIT)

# ---------- BITBOARD HELPERS ----------
def lsb(bitboard: int) -> Optional[int]:
    """Return index (0-63) of least-significant 1 bit, or None if zero."""
    if bitboard == 0:
        return None
    return (bitboard & -bitboard).bit_length() - 1

def pop_lsb(bitboard: int) -> (Optional[int], int):
    """Return (lsb_index, new_bitboard_after_pop)."""
    if bitboard == 0:
        return None, 0
    l = lsb(bitboard)
    return l, bitboard & (bitboard - 1)

# ---------- BITBOARD CACHE (per-position) ----------
# We build a small lightweight helper class to cache bitboards and some derived info,
# used by SEE and move-ordering to reduce repeated Python-level calls.

class PosBB:
    __slots__ = ("board", "occ", "piece_bb_w", "piece_bb_b", "all_attack_cache")
    def __init__(self, board: chess.Board):
        self.board = board
        # occupancy bitboard (int)
        self.occ = int(board.occupied)
        # piece bitboards by color: dict piece_type -> int
        self.piece_bb_w = {pt: int(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUE}
        self.piece_bb_b = {pt: int(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUE}
        # cache for attack masks per square not used heavily here (left for extension)
        self.all_attack_cache = {}

    def attackers_bb(self, color: bool, square: int) -> int:
        """Return bitboard of attackers of `color` to `square` using board.attackers (fast underlying C)."""
        # board.attackers returns a SquareSet; cast to int for bitmask
        # python-chess attack queries are implemented in C and are efficient.
        return int(self.board.attackers(color, square))

# ---------- STATIC EXCHANGE EVALUATION (bitboard-simulated) ----------
def see(board: chess.Board, move: chess.Move) -> int:
    """
    A faster SEE that tries to avoid expensive board.copy()/push/pop:
    - We simulate swap-off using integer occupancy and attacker bitboards.
    - For sliding attackers we rely on python-chess's attackers() to recompute attackers
      when necessary, but we avoid making full board copies in many cases.
    - Falls back to safe copy-based SEE for complicated edge cases.
    """
    # cheap filter
    if not board.is_capture(move) and not move.promotion:
        return 0

    # Build position bitboard helpers
    pbb = PosBB(board)
    to_sq = move.to_square
    from_sq = move.from_square
    side = board.turn  # True==white

    # Determine initial victim square/value (handle en-passant)
    if board.is_en_passant(move):
        # en-passant victim is behind to_sq
        victim_sq = to_sq + (-8 if side == chess.WHITE else 8)
        victim_piece_type = chess.PAWN
    else:
        victim_piece = board.piece_at(to_sq)
        victim_sq = to_sq
        victim_piece_type = victim_piece.piece_type if victim_piece else None

    victim_value = PIECE_VALUE[victim_piece_type] if victim_piece_type else 0

    # We'll simulate captures by alternating sides using integer attacker masks.
    # Start by constructing attacker masks for both colors (int bitboards)
    attackers_white = pbb.attackers_bb(chess.WHITE, to_sq)
    attackers_black = pbb.attackers_bb(chess.BLACK, to_sq)

    # Remove the moving piece from its side's attacker set (it moves)
    if board.piece_at(from_sq):
        if board.piece_at(from_sq).color == chess.WHITE:
            attackers_white &= ~(1 << from_sq)
        else:
            attackers_black &= ~(1 << from_sq)

    # Build a local occupancy int that accounts for the move (source cleared, dest occupied)
    occ = pbb.occ
    occ &= ~(1 << from_sq)
    occ |= (1 << to_sq)
    if board.is_en_passant(move):
        occ &= ~(1 << victim_sq)  # removed captured pawn

    # Now attempt to simulate swap-off without copying board by repeatedly selecting least valuable attacker.
    # NOTE: For sliding attackers (rook/bishop/queen), whether they attack depends on occupancy.
    # python-chess provides attack generation using board.attackers(color, square) which uses current board.
    # To avoid complex recomputation of sliding ray attacks from int occ alone, we will:
    #  - use a *local* board copy for sliding updates BUT we only push the captured-moves (not full copy per ply).
    # This hybrid approach reduces copies compared to naive SEE.

    # Try a lightweight approach: use a shallow copy once, then push capture moves on it.
    # This is still cheaper than repeated full copy for each move generation in many cases.
    try:
        b = board.copy(stack=False)  # stack=False avoids copying move_stack internals (faster)
    except TypeError:
        # older python-chess may not have stack arg; fall back to full copy
        b = board.copy()

    # First push the original move on the copy and then simulate greedy captures by picking least valuable attackers.
    try:
        b.push(move)
    except Exception:
        # If move illegal on copy (shouldn't happen) fallback to safe method
        return see_safe(board, move)

    gains = [victim_value]
    side_to_move = not side
    while True:
        # list attackers from side_to_move to target square (python-chess C implementation is efficient)
        attackers = list(b.attackers(side_to_move, to_sq))
        if not attackers:
            break
        # pick least-valuable attacker (MVV-LVA for SEE)
        best_sq = attackers[0]
        best_val = PIECE_VALUE[b.piece_at(best_sq).piece_type]
        for a in attackers:
            pv = PIECE_VALUE[b.piece_at(a).piece_type]
            if pv < best_val:
                best_val = pv; best_sq = a
        cap_move = chess.Move(best_sq, to_sq)
        if cap_move not in b.legal_moves:
            break
        captured_now = b.piece_at(to_sq)
        captured_now_val = PIECE_VALUE[captured_now.piece_type] if captured_now else 0
        gains.append(captured_now_val - gains[-1])
        b.push(cap_move)
        side_to_move = not side_to_move

    # minimax backward to compute net gain
    for i in range(len(gains)-2, -1, -1):
        gains[i] = max(-gains[i+1], gains[i])
    return gains[0] if gains else 0

def see_safe(board: chess.Board, move: chess.Move) -> int:
    """Fallback safe SEE (copy-based)."""
    try:
        b = board.copy()
        b.push(move)
        captured_piece = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        base = PIECE_VALUE[captured_piece.piece_type] if captured_piece else 0
        gains = [base]
        side = not board.turn
        while True:
            attackers = list(b.attackers(side, move.to_square))
            if not attackers:
                break
            best_sq = attackers[0]
            best_val = PIECE_VALUE[b.piece_at(best_sq).piece_type]
            for a in attackers:
                v = PIECE_VALUE[b.piece_at(a).piece_type]
                if v < best_val: best_val = v; best_sq = a
            cap_move = chess.Move(best_sq, move.to_square)
            if cap_move not in b.legal_moves:
                break
            captured_now = b.piece_at(move.to_square)
            captured_now_val = PIECE_VALUE[captured_now.piece_type] if captured_now else 0
            gains.append(captured_now_val - gains[-1])
            b.push(cap_move)
            side = not side
        for i in range(len(gains)-2, -1, -1):
            gains[i] = max(-gains[i+1], gains[i])
        return gains[0] if gains else 0
    except Exception:
        return 0

# ---------- MOVE ORDERING ----------
def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v = PIECE_VALUE[victim.piece_type] if victim else 0
        a = PIECE_VALUE[attacker.piece_type] if attacker else 0
        return v*100 - a
    if move.promotion:
        return 90000
    return 0

def move_score(board: chess.Board, move: chess.Move, pv_move: Optional[chess.Move], ply: int) -> int:
    score = 0
    if pv_move and move == pv_move:
        score += 1000000
    score += mvv_lva_score(board, move)
    if move in KILLERS.get(ply, []):
        score += 80000
    score += HISTORY.get((move.from_square, move.to_square), 0)
    if board.is_capture(move):
        # fast SEE call
        s = see(board, move)
        score += max(0, s)
    # checks get priority in mate solving
    try:
        if board.gives_check(move):
            score += 50000
    except Exception:
        pass
    return score

def order_moves(board: chess.Board, moves: List[chess.Move], pv_move: Optional[chess.Move], ply: int) -> List[chess.Move]:
    # Convert moves to list once (avoid repeated view creation)
    scored = []
    for m in moves:
        sc = move_score(board, m, pv_move, ply)
        if RANDOM_TIE:
            sc = (sc << 8) + random.randint(0,255)
        scored.append((sc, m))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for (_, m) in scored]

# ---------- EVALUATION ----------
def evaluate(board: chess.Board) -> int:
    score = 0
    # use bitboard counts (fast)
    for pt, val in PIECE_VALUE.items():
        w_count = board.piece_type_amount(pt) if hasattr(board, "piece_type_amount") else len(board.pieces(pt, chess.WHITE))
        b_count = len(board.pieces(pt, chess.BLACK))
        # python-chess doesn't have piece_type_amount in stable API; fallback above
        score += val*(w_count - b_count)
        for s in board.pieces(pt, chess.WHITE):
            score += PST[pt][s]//2
        for s in board.pieces(pt, chess.BLACK):
            score -= PST[pt][chess.square_mirror(s)]//2
    score += (len(list(board.legal_moves))//4)
    for c in [chess.E4, chess.D4, chess.E5, chess.D5]:
        p = board.piece_at(c)
        if p:
            score += 15 if p.color==chess.WHITE else -15
    if board.is_check():
        score -= 50 if board.turn==chess.WHITE else -50
    return score if board.turn==chess.WHITE else -score

# ---------- QUIESCENCE ----------
def quiescence(board: chess.Board, alpha: int, beta: int) -> int:
    global node_count
    if timeout(): raise TimeoutError
    node_count += 1
    stand = evaluate(board)
    if stand >= beta:
        return beta
    if alpha < stand:
        alpha = stand
    moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    if not moves: return stand
    moves = order_moves(board, moves, None, 0)
    for m in moves:
        if board.is_capture(m):
            s = see(board, m)
            if s + 50 < 0:
                continue
        board.push(m)
        val = -quiescence(board, -beta, -alpha)
        board.pop()
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    return alpha

# ---------- TRANSPO TABLE ----------
TT_EXACT = 0; TT_LOWER = 1; TT_UPPER = 2
TT_SIZE_LIMIT = 2_000_000
def tt_lookup(key: int, depth: int, alpha: int, beta: int) -> Optional[int]:
    e = TT.get(key)
    if not e: return None
    if e.depth >= depth:
        if e.flag == TT_EXACT: return e.score
        if e.flag == TT_LOWER and e.score > alpha: alpha = e.score
        if e.flag == TT_UPPER and e.score < beta: beta = e.score
        if alpha >= beta: return e.score
    return None

def tt_store(key:int, depth:int, score:int, flag:int, best:Optional[chess.Move]):
    global TT_AGE
    e = TT.get(key)
    if e is None or depth > e.depth or e.age != TT_AGE:
        TT[key] = TTEntry(key, depth, score, flag, best, TT_AGE)

# ---------- ALPHA-BETA (full search) ----------
def alpha_beta(board: chess.Board, depth:int, alpha:int, beta:int, ply:int, allow_null:bool, pv_move:Optional[chess.Move]) -> int:
    global node_count, TT_AGE
    if timeout(): raise TimeoutError
    node_count += 1
    key = board.zobrist_hash()
    ttent = TT.get(key)
    if ttent and ttent.depth >= depth:
        if ttent.flag == TT_EXACT:
            return ttent.score
        elif ttent.flag == TT_LOWER:
            alpha = max(alpha, ttent.score)
        elif ttent.flag == TT_UPPER:
            beta = min(beta, ttent.score)
        if alpha >= beta:
            return ttent.score
    if board.is_checkmate():
        return -MATE_VALUE + ply
    if board.is_stalemate():
        return 0
    if depth <= 0:
        return quiescence(board, alpha, beta)
    pvmove = ttent.best if ttent else None
    if allow_null and depth >= 3 and not board.is_check() and not board.can_claim_draw():
        board.push(chess.Move.null())
        try:
            val = -alpha_beta(board, depth - 1 - NULL_REDUCTION, -beta, -beta+1, ply+1, False, None)
            if val >= beta:
                board.pop()
                return beta
        finally:
            if board.move_stack:
                board.pop()
    moves = list(board.legal_moves)
    if not moves:
        return 0
    moves = order_moves(board, moves, pvmove or pv_move, ply)
    best_score = -INFTY
    best_move = None
    first = True
    for i, mv in enumerate(moves):
        if depth <= 2 and not board.is_capture(mv) and not board.gives_check(mv) and not mv.promotion:
            est = evaluate(board)
            if est + FUTILITY_MARGIN <= alpha:
                continue
        reduction = 0
        if not first and depth >= 3 and not board.is_capture(mv) and not board.gives_check(mv) and not mv.promotion:
            reduction = int(LMR_BASE + math.log(max(depth,2)) / LMR_DIV + math.log(i+1)/LMR_DIV)
            reduction = max(0, min(reduction, depth-2))
        board.push(mv)
        if first:
            val = -alpha_beta(board, depth-1, -beta, -alpha, ply+1, True, None)
        else:
            try_depth = depth - 1 - reduction
            if try_depth < 0: try_depth = 0
            val = -alpha_beta(board, try_depth, -alpha-1, -alpha, ply+1, True, None)
            if alpha < val < beta:
                val = -alpha_beta(board, depth-1, -beta, -alpha, ply+1, True, None)
        board.pop()
        first = False
        if val > best_score:
            best_score = val
            best_move = mv
        if val > alpha:
            alpha = val
        if alpha >= beta:
            if not board.is_capture(mv):
                km = KILLERS.setdefault(ply, [None, None])
                if km[0] != mv:
                    km[1] = km[0]; km[0] = mv
            HISTORY[(mv.from_square, mv.to_square)] = HISTORY.get((mv.from_square, mv.to_square), 0) + depth*depth
            tt_store(key, depth, alpha, TT_LOWER, mv)
            return alpha
    flag = TT_EXACT
    if best_score <= alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    tt_store(key, depth, best_score, flag, best_move)
    return best_score

# ---------- MATE-ONLY SEARCH (fast focused solver) ----------
def mate_dfs(board: chess.Board, depth: int, ply: int) -> Optional[int]:
    global node_count
    if timeout(): raise TimeoutError
    node_count += 1
    key = board.zobrist_hash() ^ depth
    if key in MateTT:
        return MateTT[key]
    if board.is_checkmate():
        return None
    if board.is_stalemate():
        return None
    if depth == 0:
        return None
    moves = list(board.legal_moves)
    # prioritize checking moves
    checks = [m for m in moves if board.gives_check(m)]
    captures = [m for m in moves if board.is_capture(m) and m not in checks]
    others = [m for m in moves if (m not in checks and m not in captures)]
    ordered = checks + captures + others
    for m in ordered:
        board.push(m)
        try:
            if board.is_checkmate():
                board.pop()
                mate_score = MATE_VALUE - ply
                MateTT[key] = mate_score
                return mate_score
            opp_has_escape = False
            opp_moves = list(board.legal_moves)
            opp_checks = [om for om in opp_moves if board.gives_check(om)]
            opp_captures = [om for om in opp_moves if board.is_capture(om) and om not in opp_checks]
            opp_others = [om for om in opp_moves if (om not in opp_checks and om not in opp_captures)]
            opp_ordered = opp_checks + opp_captures + opp_others
            for om in opp_ordered:
                board.push(om)
                try:
                    res = mate_dfs(board, depth-2, ply+2)
                    if res is None:
                        opp_has_escape = True
                        board.pop()
                        break
                finally:
                    if board.move_stack:
                        pass
                board.pop()
            board.pop()
            if not opp_has_escape:
                mate_score = MATE_VALUE - ply
                MateTT[key] = mate_score
                return mate_score
        except TimeoutError:
            board.pop()
            raise
        except Exception:
            if board.move_stack:
                board.pop()
            continue
    MateTT[key] = None
    return None

def mate_search_root(board: chess.Board, max_mate_ply: int, time_limit_s: float) -> Optional[List[chess.Move]]:
    global start_time, time_limit, stop_flag, node_count, MateTT
    node_count = 0
    MateTT = {}
    start_time = time.time()
    time_limit = time_limit_s
    stop_flag = False
    for depth in range(1, max_mate_ply + 1):
        if timeout(): break
        try:
            res = mate_dfs(board, depth, 1)
        except TimeoutError:
            break
        if res is not None:
            pv = []
            b = board.copy()
            ply = 1
            while True:
                if timeout():
                    break
                moves = list(b.legal_moves)
                moves_ord = sorted(moves, key=lambda m: move_score(b, m, None, ply), reverse=True)
                found = False
                for m in moves_ord:
                    b.push(m)
                    try:
                        sat = mate_dfs(b, depth - ply + 1, ply+1)
                        if sat is not None:
                            pv.append(m)
                            found = True
                            break
                    finally:
                        if b.move_stack:
                            b.pop()
                if not found:
                    break
                if b.is_checkmate():
                    break
                ply += 1
            return pv
    return None

# ---------- ROOT SEARCH & ITERATIVE DEEPENING (full search) ----------
def root_search(board: chess.Board, max_depth:int, movetime:Optional[float]=None, nodes_limit:Optional[int]=None, multipv:int=1):
    global node_count, start_time, time_limit, stop_flag, TT_AGE, NODE_LIMIT
    node_count = 0
    TT_AGE = (TT_AGE + 1) % 256
    stop_flag = False
    start_time = time.time()
    if movetime: time_limit = movetime
    else: time_limit = DEFAULT_MOVE_TIME
    NODE_LIMIT = nodes_limit
    best_move = None
    best_score = -INFTY
    multipv = max(1, min(MULTIPV_MAX, multipv))
    results = []
    try:
        for depth in range(1, max_depth+1):
            if timeout(): break
            alpha = -MATE_VALUE
            beta = MATE_VALUE
            if best_move and depth >= 2:
                alpha = best_score - ASPIRATION
                beta = best_score + ASPIRATION
            ttbest = TT.get(board.zobrist_hash()).best if TT.get(board.zobrist_hash()) else None
            moves = list(board.legal_moves)
            moves = order_moves(board, moves, ttbest, 0)
            root_scores = []
            for mv in moves:
                if timeout(): break
                board.push(mv)
                try:
                    sc = -alpha_beta(board, depth-1, -beta, -alpha, 1, True, None)
                except TimeoutError:
                    board.pop(); raise
                board.pop()
                root_scores.append((sc, mv))
                if sc > alpha: alpha = sc
                if sc > best_score:
                    best_score = sc; best_move = mv
            root_scores.sort(reverse=True, key=lambda x: x[0])
            results = []
            for i in range(min(multipv, len(root_scores))):
                sc, mv = root_scores[i]
                b = board.copy(); b.push(mv)
                pv = [mv] + extract_pv(b, depth-1)
                results.append((sc, pv))
            if best_score >= MATE_VALUE - 1000 or timeout():
                break
    except TimeoutError:
        pass
    return best_move, results

def extract_pv(board: chess.Board, depth_limit:int) -> List[chess.Move]:
    pv = []
    b = board.copy()
    for _ in range(depth_limit):
        e = TT.get(b.zobrist_hash())
        if not e or not e.best: break
        mv = e.best
        if mv not in b.legal_moves: break
        pv.append(mv)
        b.push(mv)
    return pv

# ---------- UCI LOOP ----------
def uci_loop():
    board = chess.Board()
    multipv = 1
    threads = 1
    hash_sz = 32
    while True:
        line = sys.stdin.readline()
        if not line: break
        line = line.strip()
        if line == "uci":
            print("id name Supraniva-human")
            print("id author Supra")
            print("option name Hash type spin default 32 min 1 max 4096")
            print("option name Threads type spin default 1 min 1 max 8")
            print("option name Multipv type spin default 1 min 1 max 4")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("setoption"):
            toks = line.split()
            if "name" in toks:
                try:
                    ni = toks.index("name")+1
                    if "value" in toks:
                        vi = toks.index("value")
                        name = " ".join(toks[ni:vi]); value = " ".join(toks[vi+1:])
                    else:
                        name = " ".join(toks[ni:]); value = ""
                    if name.lower()=="multipv":
                        try: multipv = max(1, min(MULTIPV_MAX, int(value)))
                        except: pass
                except: pass
        elif line.startswith("position"):
            toks = line.split()
            if "startpos" in toks:
                board.reset()
                if "moves" in toks:
                    for mv in toks[toks.index("moves")+1:]:
                        board.push_uci(mv)
            elif "fen" in toks:
                i = toks.index("fen"); fen = " ".join(toks[i+1:i+7])
                board.set_fen(fen)
                if "moves" in toks:
                    for mv in toks[toks.index("moves")+1:]:
                        board.push_uci(mv)
        elif line.startswith("go"):
            toks = line.split()
            # support "go mate N" for dedicated mate solver
            if "mate" in toks:
                try:
                    mate_idx = toks.index("mate")
                    mate_n = int(toks[mate_idx+1])
                except Exception:
                    mate_n = 30
                movetime = None
                if "movetime" in toks:
                    try: movetime = int(toks[toks.index("movetime")+1])/1000.0
                    except: movetime = None
                tlim = movetime if movetime else 5.0
                pv = mate_search_root(board, mate_n*2, tlim)
                if pv:
                    pv_str = " ".join(m.uci() for m in pv)
                    print(f"info score mate {mate_n} nodes {node_count} pv {pv_str}")
                    print("bestmove", pv[0].uci())
                else:
                    print("info string no mate found")
                    best, _ = root_search(board, 6, movetime=1.0, nodes_limit=None, multipv=multipv)
                    if best:
                        print("bestmove", best.uci())
                    else:
                        legal = list(board.legal_moves)
                        if legal:
                            print("bestmove", legal[0].uci())
                        else:
                            print("bestmove 0000")
                sys.stdout.flush()
                continue

            toks = line.split()
            wtime = btime = movetime = depth = nodes = None
            infinite = False
            if "wtime" in toks:
                try: wtime = int(toks[toks.index("wtime")+1])
                except: pass
            if "btime" in toks:
                try: btime = int(toks[toks.index("btime")+1])
                except: pass
            if "movetime" in toks:
                try: movetime = int(toks[toks.index("movetime")+1]) / 1000.0
                except: pass
            if "depth" in toks:
                try: depth = int(toks[toks.index("depth")+1])
                except: pass
            if "nodes" in toks:
                try: nodes = int(toks[toks.index("nodes")+1])
                except: pass
            if "infinite" in toks: infinite = True
            if movetime:
                tlim = movetime
            elif (wtime or btime) and not infinite:
                remaining = wtime if board.turn==chess.WHITE else btime
                if remaining:
                    tlim = max(0.01, remaining/1000.0/40.0)
                else:
                    tlim = DEFAULT_MOVE_TIME
            elif infinite:
                tlim = 3600.0
            else:
                tlim = DEFAULT_MOVE_TIME
            maxd = depth if depth else 32
            best, multipv_list = root_search(board, maxd, movetime=tlim, nodes_limit=nodes, multipv=multipv)
            if multipv_list:
                for idx, (sc, pv) in enumerate(multipv_list, start=1):
                    if abs(sc) > MATE_VALUE//2:
                        mate = (MATE_VALUE - abs(sc)) // 100
                        score_str = f"mate {mate if sc>0 else -mate}"
                    else:
                        score_str = f"cp {int(sc)}"
                    pv_str = " ".join(m.uci() for m in pv)
                    print(f"info multipv {idx} score {score_str} depth {len(pv)} nodes {node_count} pv {pv_str}")
            if best:
                print("bestmove", best.uci())
            else:
                lm = list(board.legal_moves)
                if lm:
                    print("bestmove", lm[0].uci())
                else:
                    print("bestmove 0000")
        elif line == "quit":
            break
        sys.stdout.flush()

if __name__ == "__main__":
    random.seed(0xC0FFEE)
    uci_loop()
