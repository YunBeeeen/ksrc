"""기준경로와 pure pursuit.

**왜 Nav2 를 직접 안 쓰는가**
학습 루프는 16 env 병렬이라 ROS2 노드를 띄울 수가 없다.  그리고 필요한 것은
플래너가 아니라 **플래너의 출력 인터페이스**다 -- 차체좌표 lookahead 점과
cmd_vel.  그래서 같은 인터페이스를 파이썬으로 미러링한다.  나중에 이 자리에
Nav2(`nav2_mppi_controller`, motion_model=Omni)를 꽂아도 정책은 그대로 쓴다.

**왜 RPP 가 아니라 홀로노믹 추종인가**
`nav2_regulated_pure_pursuit_controller` 는 differential/ackermann 용이라
(v, omega) 만 낸다.  스워브의 게걸음(vy)을 통째로 버리는 셈이라, 여기서는
lookahead 점을 향한 속도벡터를 그대로 낸다 (vx, vy, omega).

**왜 경로를 매번 새로 뽑는가**
경로 하나에 고정하면 그 경로에 과적합된다.  그리고 직선만 주면 나중에 실제
플래너가 내는 곡선 경로에서 무너지므로, 중간점을 꺾어 곡률을 일부러 섞는다.
"""
import numpy as np

import terrain as terr


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class RefPath:
    """꺾인 선분열 + 호길이 파라미터화.

    투영은 **단조 진행**으로 한다.  전역 최근접점을 쓰면 경로가 자기 근처로
    되돌아오는 구간에서 진행도가 뒤로 점프하고, 그러면 진행 보상이 음수로
    튀어 학습이 망가진다.
    """

    def __init__(self, pts):
        self.P = np.asarray(pts, float)
        seg = np.diff(self.P, axis=0)
        L = np.linalg.norm(seg, axis=1)
        keep = L > 1e-6
        self.P = np.vstack([self.P[:1], self.P[1:][keep]])
        seg = np.diff(self.P, axis=0)
        self.L = np.linalg.norm(seg, axis=1)
        self.U = seg / self.L[:, None]
        self.S = np.concatenate([[0.0], np.cumsum(self.L)])
        self.total = float(self.S[-1])
        self.i = 0                      # 진행 중인 구간 (되돌아가지 않는다)
        self.s_last = 0.0               # 직전 진행도 (단조·상한용)
        self.p_last = self.P[0].copy()   # 직전 질의 위치 (변위 상한용)

    # ------------------------------------------------------------------
    def project(self, p, limit=True):
        """(호길이 s, 부호있는 횡이탈 e_y, 접선 단위벡터) 를 돌려준다.

        **선분 위 최근접점까지의 실제 거리**로 후보를 고른다.  예전엔 선분을
        무한히 연장한 직선까지의 수직거리로 골랐는데, 그러면 두 가지가 깨졌다:

          - 선분 끝을 지나면 연장선 위에 있어 `e_y = 0` 이 나온다.  직선 1m
            경로의 끝에서 1.02m 를 지난 점이 `e_y=0`, 남은거리 0 으로 계산돼
            **완주 판정까지 True** 였다.
          - 꺾인 경로에서 아직 도달하지 않은 뒤쪽 선분의 연장선이 더 가까우면
            거기로 넘어간다.  ㄱ자 경로의 (0.6, 0.5) 에서 실제 진행은 0.60m
            인데 `s = 1.500` 이 나왔다 -- 코너를 건너뛰면 0.9m 의 유령 진행
            보상이 붙는다.

        `e_y` 는 **최근접점까지의 부호있는 거리**다.  선분 내부에 떨어지면
        직선까지의 수직거리와 같고, 끝을 벗어나면 끝점까지의 거리가 된다
        (부호는 진행방향 기준 좌/우를 유지).

        최근접점 기준만으로는 **코너 건너뛰기**가 안 막힌다.  ㄱ자 경로의
        (0.6, 0.5) 는 선분1 까지가 0.40m, 선분0 까지가 0.50m 로 선분1 이 실제로
        더 가깝다 -- 기하적으로는 맞지만 진행도가 0.6 -> 1.5 로 튄다.

        그래서 진행도 증가를 **실제 변위 `|p - p_last|` 로 상한**한다.  경로를 따라
        간 거리가 실제로 움직인 거리를 넘을 수 없다는 물리적 사실이다.

        상한을 고정 상수(`max_adv`)로 두면 안 된다 -- 그러면 코너를 자른 위치에
        **정지해 있어도** 매 스텝 상한만큼 진행도가 올라간다 (실측: s 가 0.030씩
        증가, prog = 6.8 -> clip 1.0 = 최대 보상).  코너 자르기 악용을 정지 악용으로
        바꾸는 셈이다.
        """
        best = None
        for j in range(self.i, min(self.i + 3, len(self.L))):
            d = p - self.P[j]
            t = float(np.clip(np.dot(d, self.U[j]), 0.0, self.L[j]))
            q = self.P[j] + self.U[j] * t              # 선분 위 최근접점
            v = p - q
            dist = float(np.linalg.norm(v))
            sgn = np.sign(np.cross(self.U[j], v))
            if sgn == 0.0:
                sgn = np.sign(np.cross(self.U[j], d)) or 1.0
            if best is None or dist < best[0]:
                best = (dist, self.S[j] + t, float(sgn) * dist, j)
        _, s, e, j = best
        if limit:
            # 실제 변위가 상한.  정지하면 0 이므로 진행도가 안 올라간다.
            cap = self.s_last + float(np.linalg.norm(p - self.p_last))
            if s > cap:                       # 건너뛰기 -- 진행도를 묶는다
                s = cap
                j = int(np.clip(np.searchsorted(self.S, s, side="right") - 1,
                                0, len(self.L) - 1))
                d = p - self.P[j]
                t = float(np.clip(np.dot(d, self.U[j]), 0.0, self.L[j]))
                q = self.P[j] + self.U[j] * t
                v = p - q
                dist = float(np.linalg.norm(v))
                sg = np.sign(np.cross(self.U[j], v)) or 1.0
                e = float(sg) * dist
            s = max(s, self.s_last)           # 단조
        self.s_last = s
        self.p_last = np.asarray(p, float).copy()
        self.i = j
        return s, e, self.U[j]

    def at(self, s):
        """호길이 s 지점의 좌표 (양끝은 잘라낸다)."""
        s = float(np.clip(s, 0.0, self.total))
        j = int(np.searchsorted(self.S, s, side="right") - 1)
        j = int(np.clip(j, 0, len(self.L) - 1))
        return self.P[j] + self.U[j] * (s - self.S[j])


def make_path(drive, Z, ex, ey, rng, start, min_len=2.0, max_len=3.5,
              step=(0.7, 1.4), turn_deg=70.0, tries=40,
              slope_deg=None, d_ep=0.0, n_cand=8):
    """주행 가능 셀만 밟는 꺾인 경로.

    한 번 막혔을 때 바로 포기하면 30% 가 0.5m 직선으로 퇴화했다 (막힌 방향을
    계속 고집하기 때문).  그래서 좁은 선회각으로 먼저 시도하고, 실패하면
    **선회각 제한을 풀어** 되돌아가는 방향까지 허용한다.  초기 방향도 경기장
    안쪽으로 편향시킨다 -- 벽을 향해 시작하면 첫 구간부터 막힌다.

    `d_ep` 에 비례해 **급사면을 지나는 구간**을 고른다.  line_clear 만 보면
    경로가 험한 지형을 전부 피해가서, 난이도를 올려도 과제가 안 어려워진다
    (그 상태로 재보니 순수 IK 가 d=1.0 에서 93% 를 완주했다).  30도 사면은
    맵 면적의 1% 라 무작위로는 거의 안 지나간다.
    """
    start = np.asarray(start, float)
    pts = [start]
    head = float(np.arctan2(-start[1], -start[0]))          # 중심 방향
    head += float(rng.uniform(-np.pi / 2, np.pi / 2))
    total = 0.0
    while total < min_len:
        got = False
        for lim in (turn_deg, 170.0):                       # 2단: 좁게 -> 넓게
            cand = []
            for _ in range(tries):
                r = float(rng.uniform(*step))
                a = head + float(np.radians(rng.uniform(-lim, lim)))
                nxt = pts[-1] + r * np.array([np.cos(a), np.sin(a)])
                if abs(nxt[0]) > ex / 2 - 0.3 or abs(nxt[1]) > ey / 2 - 0.3:
                    continue
                if not terr.line_clear(drive, ex, ey, pts[-1], nxt):
                    continue
                cand.append((nxt, a, r))
                if len(cand) >= n_cand:
                    break
            if not cand:
                continue
            if slope_deg is not None and rng.random() < d_ep:
                j = int(np.argmax([terr.line_mean(slope_deg, ex, ey, pts[-1], cc[0])
                                   for cc in cand]))
            else:
                j = 0
            nxt, a, r = cand[j]
            pts.append(nxt); head = a; total += r; got = True
            break
        if not got:
            break
        if total > max_len:
            break
    if len(pts) < 2:                    # 완전 실패 -- 전방 0.5m 직선
        pts.append(pts[0] + 0.5 * np.array([np.cos(head), np.sin(head)]))
    return RefPath(pts)


def pursue(path, s, pos, yaw, v_cruise, k_v=2.0, L_min=0.25, L_max=0.80,
           k_psi=1.2, om_lim=0.8):
    """차체좌표 속도명령 (vx, vy, omega).

    lookahead 거리를 속도에 비례시키는 것(regulated)은 Nav2 RPP 와 같다.
    요는 **경로 방향에 정렬**시킨다 -- 스워브는 요가 자유도지만, 명령을 주지
    않으면 슬립으로 틀어진 요를 되잡을 학습 신호가 생기지 않는다.

    요 기준을 **선분 접선이 아니라 [s, s+L] 구간의 현(chord)** 으로 잡는다.
    접선은 꼭짓점에서 최대 70도 튀어서 omega 명령에 계단이 생기고(측정: 중앙
    0.584 rad/s, 13% 가 ±0.8 포화), 정책이 물리적으로 못 따라갈 것을 벌한다.
    현 방향은 s 에 대해 연속이고 직선 구간에서는 접선과 같다.
    또한 현은 **경로만의 함수**라 횡이탈과 커플링되지 않는다 -- 로버에서 본
    lookahead 방향을 요 기준으로 쓰면, 경로를 벗어날수록 몸을 틀어버려서
    vy(게걸음)로 복귀하는 경로가 막힌다.
    """
    L = float(np.clip(k_v * v_cruise, L_min, L_max))
    tgt = path.at(s + L)
    d = tgt - pos
    n = float(np.linalg.norm(d))
    rem = path.total - s
    v_w = d / max(n, 1e-6) * min(v_cruise, 1.5 * max(rem, 0.0))
    c, sn = np.cos(yaw), np.sin(yaw)
    v_b = np.array([c * v_w[0] + sn * v_w[1], -sn * v_w[0] + c * v_w[1]])
    chord = path.at(s + L) - path.at(s)
    if float(np.linalg.norm(chord)) < 1e-6:
        chord = path.U[path.i]
    om = float(np.clip(k_psi * _wrap(np.arctan2(chord[1], chord[0]) - yaw),
                       -om_lim, om_lim))
    return np.array([v_b[0], v_b[1], om])


def lookahead_body(path, s, pos, yaw, dists=(0.30, 0.80)):
    """관측용: 앞쪽 lookahead 점들을 차체좌표로.

    실기에서는 Nav2 의 local plan + TF 에서 나온다.  즉 **측위가 있어야**
    얻을 수 있는 값이므로, env 에서 측위 오차를 노이즈로 섞어 준다.
    """
    c, sn = np.cos(yaw), np.sin(yaw)
    out = []
    for L in dists:
        d = path.at(s + L) - pos
        out.append([c * d[0] + sn * d[1], -sn * d[0] + c * d[1]])
    return np.asarray(out, float)


def start_heading(path, L=0.5):
    """경로 시작부의 진행 방향.  스폰 자세를 여기에 대충 맞춘다."""
    d = path.at(min(L, path.total)) - path.P[0]
    return float(np.arctan2(d[1], d[0]))


def heading_at(path, s, eps=0.05):
    """호길이 s 지점의 **경로 방향**.  중앙 현(central chord) 으로 정의한다.

    선분 접선을 그대로 쓰면 꼭짓점에서 각도가 순간 점프한다 (우리 경로는 꺾임이
    최대 70도).  그러면 스폰 자세·관측 e_psi·보상이 같은 지점에서 불연속을 겪는다.
    중앙 현은 s 에 대해 연속이고 직선 구간에서는 접선과 같다.

    스폰 자세·관측 e_psi·실제 이동방향 보상이 같은 접선 정의를 공유한다.
    """
    a = path.at(max(s - eps, 0.0))
    b = path.at(min(s + eps, path.total))
    d = b - a
    if float(np.linalg.norm(d)) < 1e-9:          # 경로가 eps 보다 짧은 극단
        d = path.U[min(path.i, len(path.U) - 1)]
    return float(np.arctan2(d[1], d[0]))


def make_path_set(n, drive, Z, ex, ey, seed, **kw):
    """고정 경로 세트.  **학습용과 평가용을 반드시 다른 seed 로 만든다.**

    같은 세트로 학습·평가하면 `s0` 를 랜덤화해도 "그 geometry 를 외운 건지"
    구분할 수 없다.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        st = terr.sample_drivable(drive, Z, ex, ey, rng, 1)[0]
        pth = make_path(drive, Z, ex, ey, rng, st, **kw)
        if pth.total >= 2.0:                     # 퇴화 경로 제외
            out.append(pth)
    return out


def clone(path):
    """같은 꼭짓점의 새 RefPath.  투영 상태(s_last, p_last, i)가 독립이어야
    보상용(참값)과 컨트롤러용(추정치)을 분리할 수 있다."""
    return RefPath(path.P)
