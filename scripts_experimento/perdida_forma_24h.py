# -*- coding: utf-8 -*-
"""
¿Una funcion de perdida que premie la forma reduce el amortiguamiento? (prueba del 6-oct-2026)

Mismo modelo, cuatro perdidas: una red pequeña (MLP) que corrige la forma del ensamble a partir de las 24 h
de los 5 votantes y del ensamble (todo relativo a la mediana del dia pronosticada), con calendario y nivel.
  * L1               error absoluto (lo de siempre)
  * L1 + cambios     L1 + L1 de los cambios hora a hora (premia acertar la magnitud de cada cambio)
  * DILATE           Le Guen y Thome (NeurIPS 2019): soft-DTW (forma) + indice de distorsion temporal
  * DILATE + L1      DILATE con un termino L1 para no perder el nivel
Entrena con los origenes historicos 1-5 (2020-2025; 10 % final para parada temprana) y evalua en 2026
(ventanas de 24 h), 3 semillas por perdida.
Metricas: MAE, amplitud (sd del pronostico / sd real dentro del dia), rampas de 3 h (direccion y fraccion
de la magnitud anticipada) y la descomposicion del error cuadratico (nivel / amplitud / forma).
Salida: data/processed/resultados/perdida_forma_24h.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from eventos_escenarios_24h import panel, VOT  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "data" / "processed" / "resultados"
torch.set_num_threads(4)


def tensores(d):
    canales = VOT + ["q50"]
    filas = []
    for (o, v), g in d.groupby(["origen", "ventana"], sort=True):
        m = float(np.median(g["q50"]))
        x = np.concatenate([g[c].to_numpy() / m for c in canales])
        cal = np.zeros(7 + 3)
        cal[v.dayofweek] = 1
        cal[7], cal[8] = np.sin(2 * np.pi * v.month / 12), np.cos(2 * np.pi * v.month / 12)
        cal[9] = np.log(m) / 7
        filas.append((o, v, m, np.concatenate([x, cal]), g["real"].to_numpy() / m, g["q50"].to_numpy() / m))
    o, v, m, X, Y, E = zip(*filas)
    return pd.DataFrame({"origen": o, "ventana": v, "m": m}), np.array(X, np.float32), np.array(Y, np.float32), np.array(E, np.float32)


class Red(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 24))

    def forward(self, x, ens):
        return ens + 0.5 * self.f(x)          # parte del ensamble y aprende la correccion de forma


def soft_dtw(D, gamma):
    """Soft-DTW (Cuturi y Blondel, 2017) por programacion dinamica; D: (B, n, n) distancias."""
    B, n, _ = D.shape
    inf = torch.full((B,), 1e10, dtype=D.dtype)
    R = [[inf] * (n + 1) for _ in range(n + 1)]
    R[0][0] = torch.zeros(B, dtype=D.dtype)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            r = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]], dim=1)
            R[i][j] = D[:, i - 1, j - 1] - gamma * torch.logsumexp(-r / gamma, dim=1)
    return R[n][n]


def dilate(pred, y, alpha=0.5, gamma=0.01):
    n = y.shape[1]
    D = (y[:, :, None] - pred[:, None, :]) ** 2
    forma = soft_dtw(D, gamma)
    camino = torch.autograd.grad(forma.sum(), D, create_graph=True)[0]      # alineamiento esperado
    idx = torch.arange(n, dtype=pred.dtype)
    omega = (idx[:, None] - idx[None, :]) ** 2 / n ** 2
    tiempo = (camino * omega).sum(dim=(1, 2))
    return (alpha * forma + (1 - alpha) * tiempo).mean()


def perdida(nombre, pred, y):
    l1 = (pred - y).abs().mean()
    if nombre == "L1":
        return l1
    if nombre == "L1 + cambios":
        return l1 + (torch.diff(pred, dim=1) - torch.diff(y, dim=1)).abs().mean()
    if nombre == "DILATE":
        return dilate(pred, y)
    if nombre == "DILATE + L1":
        return dilate(pred, y) + l1
    raise ValueError(nombre)


def entrenar(nombre, Xtr, Ytr, Etr, Xva, Yva, Eva, semilla, epocas=80):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    red = Red(Xtr.shape[1])
    opt = torch.optim.Adam(red.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr, Ytr, Etr = map(torch.from_numpy, (Xtr, Ytr, Etr))
    Xva, Yva, Eva = map(torch.from_numpy, (Xva, Yva, Eva))
    mejor, estado, paciencia = np.inf, None, 0
    for ep in range(epocas):
        red.train()
        orden = torch.randperm(len(Xtr))
        for k in range(0, len(orden), 64):
            b = orden[k:k + 64]
            opt.zero_grad()
            perdida(nombre, red(Xtr[b], Etr[b]), Ytr[b]).backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            val = (red(Xva, Eva) - Yva).abs().mean().item() if nombre.startswith("L1") else None
        if val is None:
            val = perdida(nombre, red(Xva, Eva), Yva).item()
        if val < mejor - 1e-5:
            mejor, estado, paciencia = val, {k: v.clone() for k, v in red.state_dict().items()}, 0
        else:
            paciencia += 1
            if paciencia >= 10:
                break
    red.load_state_dict(estado)
    return red


def evaluar(P, Y, M):
    """P, Y relativos (n_dias, 24); M nivel por dia. Devuelve metricas en COP/kWh."""
    p, y = P * M[:, None], Y * M[:, None]
    sp, sy = p.std(axis=1), y.std(axis=1)
    rho = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(p, y)])
    nivel = ((p.mean(1) - y.mean(1)) ** 2).sum()
    amp = ((sp - sy) ** 2).sum()
    forma = np.nansum(2 * sp * sy * (1 - rho))
    tot = nivel + amp + forma
    dr, dp = y[:, 3:] - y[:, :-3], p[:, 3:] - p[:, :-3]
    dr, dp = dr.ravel(), dp.ravel()
    top = np.abs(dr) >= np.quantile(np.abs(dr), .9)
    return {"MAE": np.abs(p - y).mean(), "amplitud": np.median(sp / sy),
            "rampa_dir_%": (np.sign(dr[top]) == np.sign(dp[top])).mean() * 100,
            "rampa_fraccion": np.median(dp[top] / dr[top]),
            "err_nivel_%": nivel / tot * 100, "err_amplitud_%": amp / tot * 100, "err_forma_%": forma / tot * 100}


if __name__ == "__main__":
    t0 = time.time()
    info, X, Y, E = tensores(panel())
    te = (info["origen"] == "Origen 6").to_numpy()
    hist = np.where(~te)[0]
    va = hist[int(len(hist) * .9):]
    tr = hist[:int(len(hist) * .9)]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    M = info["m"].to_numpy()
    filas = [{"perdida": "ensamble (referencia)", "semilla": "-", **evaluar(E[te], Y[te], M[te])}]
    for nombre in ("L1", "L1 + cambios", "DILATE", "DILATE + L1"):
        for semilla in (0, 1, 2):
            red = entrenar(nombre, Xn[tr], Y[tr], E[tr], Xn[va], Y[va], E[va], semilla)
            with torch.no_grad():
                P = red(torch.from_numpy(Xn[te]), torch.from_numpy(E[te])).numpy()
            filas.append({"perdida": nombre, "semilla": semilla, **evaluar(P, Y[te], M[te])})
            print(f"{nombre} semilla {semilla}: MAE {filas[-1]['MAE']:.2f} amp {filas[-1]['amplitud']:.2f} "
                  f"rampa {filas[-1]['rampa_fraccion']:.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(RES / "perdida_forma_24h.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.groupby("perdida", sort=False).mean(numeric_only=True).round(2).to_string())
