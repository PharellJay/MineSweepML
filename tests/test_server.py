"""The web API, against a real server process (python -m minesweepml serve) on a free port."""

import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image

from minesweepml.network import MineNet, NetConfig, save_checkpoint
from minesweepml.runs import create_bot


class Response:
    def __init__(self, status, headers, body):
        self.status_code, self.headers, self.content = status, {k.lower(): v for k, v in headers.items()}, body

    @property
    def text(self):
        return self.content.decode()

    def json(self):
        return json.loads(self.content)


class Client:
    def __init__(self, base):
        self.base = base

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return Response(r.status, dict(r.headers), r.read())
        except urllib.error.HTTPError as e:
            return Response(e.code, dict(e.headers), e.read())

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, json=None):
        return self.request("POST", path, json if json is not None else {})

    def patch(self, path, json=None):
        return self.request("PATCH", path, json)

    def delete(self, path):
        return self.request("DELETE", path)


@pytest.fixture
def client(isolated_runs):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "MINESWEEPML_RUNS": str(isolated_runs)}
    proc = subprocess.Popen([sys.executable, "-m", "minesweepml", "serve", "--no-browser", "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    c = Client(f"http://127.0.0.1:{port}")
    try:
        for _ in range(100):
            try:
                if c.get("/api/info").status_code == 200:
                    break
            except OSError:
                time.sleep(0.1)
        yield c
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture
def bot():
    path = create_bot("Tester", {"blocks": 2, "channels": 8})
    save_checkpoint(str(path / "checkpoints" / "ckpt_0000000g.pt"), MineNet(NetConfig(2, 8)), games=0)
    return path


def test_index_and_info(client):
    r = client.get("/")
    assert r.status_code == 200 and "MineSweepML" in r.text
    assert r.headers["cache-control"] == "no-cache"
    info = client.get("/api/info").json()
    assert set(info["presets"]) == {"light", "balanced", "heavy", "hardcore"}
    assert info["difficulties"]["expert"]["mines"] == 99


def test_bot_crud(client):
    r = client.post("/api/bots", json={"name": "Alpha", "settings": {"blocks": 2, "channels": 8, "mix_expert": 1}})
    assert r.status_code == 200
    assert client.post("/api/bots", json={"name": "Alpha"}).status_code == 409
    assert client.post("/api/bots", json={"name": "bad/name"}).status_code == 400
    r = client.patch("/api/bots/Alpha", json={"name": "Beta", "settings": {"lr": 0.0005}})
    assert r.json() == {"name": "Beta", "settings": {**r.json()["settings"], "lr": 0.0005}}
    assert client.patch("/api/bots/Beta", json={"settings": {"channels": 64}}).status_code == 400
    assert [b["name"] for b in client.get("/api/bots").json()] == ["Beta"]
    assert client.delete("/api/bots/Beta").status_code == 200
    assert client.get("/api/bots").json() == []
    assert client.delete("/api/bots/Beta").status_code == 404


def test_play_replay_probabilities_and_gif(client, bot):
    label = "Tester/ckpt_0000000g.pt"
    res = client.post("/api/simulate/game", json={"player": label, "difficulty": "beginner", "seed": 3}).json()
    record, replay = res["record"], res["replay"]
    assert replay["rows"] == 9 and len(replay["opened"]) == len(record["moves"])
    assert client.post("/api/replay", json={"record": record}).json()["moves"] == replay["moves"]

    probs = client.post("/api/replay/probabilities", json={"record": record, "player": label}).json()["probs"]
    assert len(probs) == len(record["moves"]) and len(probs[0]) == 81
    solver = client.post("/api/replay/probabilities", json={"record": record, "player": "solver"}).json()["probs"]
    assert all(-1 <= p <= 100 for row in solver for p in row)

    r = client.post("/api/replay/gif", json={"record": record, "player": label, "fps": 10})
    assert r.status_code == 200 and r.headers["content-type"] == "image/gif"
    img = Image.open(io.BytesIO(r.content))
    assert img.n_frames == len(record["moves"]) + 1

    assert client.post("/api/simulate/game", json={"player": "nobody", "difficulty": "beginner"}).status_code == 400
    assert client.post("/api/simulate/game", json={"player": "solver", "difficulty": "99x99/5"}).status_code == 400


def test_training_rejects_bad_input(client):
    assert client.post("/api/training/start", json={"duration": "time", "time": "soon"}).status_code == 400
    assert client.post("/api/training/start", json={"workers": 0}).status_code == 400
    assert client.post("/api/benchmarks", json={"player": "solver", "boards": 5}).status_code == 400
