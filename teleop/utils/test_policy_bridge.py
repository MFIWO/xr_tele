"""Loopback check for the policy bridge client. No robot and no GPU needed.

Runs PolicyBridgeClient against a stub server that speaks the same wire format and
echoes the observed state back as the action chunk, then asserts the control-loop
contract: the handshake settles, one action pops per tick, steps never rewind, and
observations are only sent when the queue runs low.

    python teleop/utils/test_policy_bridge.py
"""

import os
import socket
import socketserver
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from teleop.utils.policy_bridge import PolicyBridgeClient, decode_frame, encode_frame

CAMERAS = ["head_left", "head_right", "wrist_left", "wrist_right"]
IMAGE_SIZE = (224, 224)
CHUNK = 20
FPS = 20.0
STATE_DIM = 42


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            meta, _ = decode_frame(self.request)
            assert meta["type"] == "hello", meta
            encode_frame(self.request, {
                "type": "ready",
                "policy": "echo-state",
                "device": "cpu",
                "cameras": CAMERAS,
                "image_size": list(IMAGE_SIZE),
                "state_dim": STATE_DIM,
                "action_dim": STATE_DIM,
                "chunk_size": CHUNK,
            })
            while True:
                meta, payload = decode_frame(self.request)
                if meta.get("type") != "obs":
                    continue
                self.server.observations += 1
                assert len(meta["state"]) == STATE_DIM
                assert [c["name"] for c in meta["cameras"]] == CAMERAS
                assert sum(c["len"] for c in meta["cameras"]) == len(payload)
                chunk = np.repeat(
                    np.asarray(meta["state"], dtype="<f4")[None, :], CHUNK, axis=0
                )
                encode_frame(self.request, {
                    "type": "act",
                    "seq": meta["seq"],
                    "t_obs": meta["t"],
                    "n": CHUNK,
                    "dim": STATE_DIM,
                    "infer_ms": 0.0,
                }, np.ascontiguousarray(chunk).tobytes())
        except (OSError, ConnectionError, AssertionError, KeyError):
            return


class EchoServer(socketserver.TCPServer):
    allow_reuse_address = True
    observations = 0


def main() -> None:
    server = EchoServer(("127.0.0.1", 0), EchoHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    client = PolicyBridgeClient(
        addr=f"127.0.0.1:{port}",
        task="loopback",
        cameras=CAMERAS,
        state_dim=STATE_DIM,
        action_dim=STATE_DIM,
        fps=FPS,
        image_encoding="raw",
    )
    client.start()
    try:
        deadline = time.time() + 5.0
        while not client.ready and time.time() < deadline:
            time.sleep(0.02)
        assert client.ready, "handshake never completed"
        assert client.image_size == IMAGE_SIZE, client.image_size
        assert client.chunk_size == CHUNK, client.chunk_size

        frames = {cam: np.full((*IMAGE_SIZE, 3), 40 * i % 255, dtype=np.uint8)
                  for i, cam in enumerate(CAMERAS)}
        state = np.arange(STATE_DIM, dtype=np.float32)

        ticks = 60
        executed = []
        for _ in range(ticks):
            client.submit(state, frames)
            action = client.pop_action()
            if action is not None:
                executed.append(action)
            time.sleep(1.0 / FPS)

        stats = client.stats()
        assert len(executed) >= ticks - 5, f"only {len(executed)} of {ticks} ticks got an action"
        assert stats["step"] == len(executed), (stats["step"], len(executed))
        assert stats["dropped"] == 0, stats
        # chunk_size_threshold=0.5 over a 20-action chunk: roughly one observation per
        # 10 executed actions, never one per tick.
        assert stats["sent"] <= ticks // 5, f"sent {stats['sent']} observations in {ticks} ticks"
        assert np.allclose(executed[-1], state, atol=1e-5), executed[-1][:5]

        print(f"OK: {len(executed)}/{ticks} ticks acted, {stats['sent']} observations sent, "
              f"rtt={stats['rtt_ms']:.1f}ms")
    finally:
        client.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
